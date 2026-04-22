import threading
import numpy as np
import g2opy as g2o
import cv2 as cv

import Map
import Params

class Backend:
    def __init__(self, params_stereo, map : Map):
        self.map_ = map
        self.params_stero_ = params_stereo
        
        self.K = self.params_stero_['P1'][:3, :3]
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.baseline = -self.params_stero_['P2'][0, 3] / self.fx

        self.MP_OFFSET = 100000

        self.loop_constraints = []
        self._loop_mutex = threading.Lock()

    def add_loop_constraints(self, loops):
        """loops: list of (kf_query_id, kf_candidate_id). The relative pose is
        computed now, from the current (noisy) frontend poses, so the edge
        snapshots the measurement at detection time."""
        all_kfs = {kf.id_: kf for kf in self.map_.get_all_keyframes()}
        new_constraints = []
        for query_id, cand_id in loops:
            kf_q = all_kfs.get(query_id)
            kf_c = all_kfs.get(cand_id)
            if kf_q is None or kf_c is None:
                continue
            # Pose vertices store T_cw as SE3Quat. For EdgeSE3Expmap with
            # v1=candidate, v2=query, the measurement is T_v2 * T_v1^-1.
            T_rel = kf_q.pose_ @ np.linalg.inv(kf_c.pose_)
            new_constraints.append({
                "kf1": cand_id,
                "kf2": query_id,
                "T_rel": T_rel,
                "info": np.eye(6) * 100.0,
            })
        with self._loop_mutex:
            self.loop_constraints.extend(new_constraints)

    def setup_optimizer(self):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(algorithm)
        
        cam_params = g2o.CameraParameters(self.fx, np.array([self.cx, self.cy]), self.baseline)
        cam_params.set_id(0)
        optimizer.add_parameter(cam_params)
        return optimizer

    def update_map(self):
        active_keyframes = self.map_.get_active_keyframes()
        active_mappoints = self.map_.get_active_map_points()
        
        if len(active_keyframes) < 3:
            return
            
        optimizer = self.setup_optimizer()
        
        # Vertices - Poses
        for kf in active_keyframes:
            se3 = g2o.SE3Quat(kf.pose[:3, :3], kf.pose[:3, 3])
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(kf.id_)
            v_se3.set_estimate(se3)
            if kf.id_ == active_keyframes[0].id_:
                v_se3.set_fixed(True)
            optimizer.add_vertex(v_se3)

        # Vertices - Landmarks
        for mp in active_mappoints:
            v_p = g2o.VertexPointXYZ()
            v_p.set_id(mp.id_ + self.MP_OFFSET)
            v_p.set_estimate(mp.pos)
            v_p.set_marginalized(True) 
            optimizer.add_vertex(v_p)

        # Edges/observations
        chi2_th = Params.get('chi2_threshold')
        edges_and_features = []
        valid_kf_ids = [kf.id_ for kf in active_keyframes]

        for mp in active_mappoints:
            for obs_ref in mp.get_obs():
                feature = obs_ref()
                if feature is None or feature.frame_() is None or feature.is_outlier_:
                    continue
                    
                kf = feature.frame_()
                if kf.id_ not in valid_kf_ids:
                    continue 
                
                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, optimizer.vertex(mp.id_ + self.MP_OFFSET))
                edge.set_vertex(1, optimizer.vertex(kf.id_))
                edge.set_measurement(np.array(feature.position_.pt))
                edge.set_information(np.eye(2))
                edge.set_parameter_id(0, 0)
                
                rk = g2o.RobustKernelHuber()
                rk.set_delta(np.sqrt(chi2_th))
                edge.set_robust_kernel(rk)
                
                optimizer.add_edge(edge)
                edges_and_features.append((edge, feature, mp))

        # Loop-closure edges (only usable if both endpoints are in the window)
        with self._loop_mutex:
            pending = self.loop_constraints
            self.loop_constraints = []
        loop_edges_added = 0
        for c in pending:
            if c["kf1"] not in valid_kf_ids or c["kf2"] not in valid_kf_ids:
                continue
            edge = g2o.EdgeSE3Expmap()
            edge.set_vertex(0, optimizer.vertex(c["kf1"]))
            edge.set_vertex(1, optimizer.vertex(c["kf2"]))
            T = c["T_rel"]
            edge.set_measurement(g2o.SE3Quat(T[:3, :3], T[:3, 3]))
            edge.set_information(c["info"])
            rk = g2o.RobustKernelHuber()
            rk.set_delta(np.sqrt(7.815))
            edge.set_robust_kernel(rk)
            optimizer.add_edge(edge)
            loop_edges_added += 1
        if loop_edges_added > 0:
            print(f"[BACKEND] {loop_edges_added} loop-closure edge(s) added to BA")

        # Optimisation
        optimizer.initialize_optimization()
        optimizer.optimize(10)

        # Rejet des outliers - on continue jusqu'à avoir +50% d'outliers ou itérer 5 fois
        iteration = 0
        while iteration < 5:
            cnt_outlier = 0
            cnt_inlier = 0
            for edge, feat, mp in edges_and_features:
                if edge.chi2() > chi2_th:
                    cnt_outlier += 1
                else:
                    cnt_inlier += 1
            
            inlier_ratio = cnt_inlier / max(1, (cnt_inlier + cnt_outlier))
            if inlier_ratio > 0.5:
                break
            else:
                chi2_th *= 2.0
                iteration += 1

        # Application des outliers
        for edge, feat, mp in edges_and_features:
            if edge.chi2() > chi2_th:
                feat.is_outlier_ = True
                # On détache le point de la carte
                mp.remove_observation(feat) 
            else:
                feat.is_outlier_ = False

        print(f"[BACKEND] Outliers/Inliers : {cnt_outlier} / {cnt_inlier}")

        # mise à jour de la carte
        for kf in active_keyframes:
            if kf.id_ == active_keyframes[0].id_: continue
            v_se3 = optimizer.vertex(kf.id_)
            kf.pose = v_se3.estimate().matrix()
            
        for mp in active_mappoints:
            v_p = optimizer.vertex(mp.id_ + self.MP_OFFSET)
            mp.pos = v_p.estimate()
            
        # nettoyage de la mémoire
        self.map_.clean_map()