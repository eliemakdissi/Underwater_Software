import threading
import cv2 as cv
import numpy as np
import time

import Params
from Frame import Frame
from Map import Map
from MapPoint import MapPoint
from Backend import Backend

class FrontendStatus():
    INITING = 0
    TRACKING_GOOD = 1
    TRACKING_BAD = 2
    LOST = 3

class Frontend():

    def __init__(self, params_stereo_, map: Map, backend: Backend):
        self.status_ = FrontendStatus.INITING
        self.params_stero_ = params_stereo_
        
        self.current_frame_ = None
        self.previous_frame_ = None
        self.last_keyframe_ = None
        self.num_frames_since_last_kf_ = 0
        
        self.map_ = map
        self.backend_ = backend
        
        self.matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

        # Filtre d'outliers
        self.minimal_number_neigbors = 15
        self.radius = 0.1

    def add_frame(self, frame: Frame):
        self.current_frame_ = frame
        success = False 

        if self.status_ == FrontendStatus.INITING:
            success = self.stero_init()

        elif self.status_ in [FrontendStatus.TRACKING_GOOD, FrontendStatus.TRACKING_BAD]:
            success = self.tracking()

        elif self.status_ == FrontendStatus.LOST:
            self.reset()
            success = self.stero_init()

        self.previous_frame_ = self.current_frame_
        return success

    def stero_init(self): 
        self.current_frame_.preprocess()
        t1 = time.time()
        self.current_frame_.extract_features()
        t2 = time.time()
        features_l = self.current_frame_.features_left_
        features_r = self.current_frame_.features_right_

        if not features_l or not features_r:
            return False

        # --- 1. MATCHING GAUCHE/DROITE ---
        desc_l = np.array([f.descriptor_ for f in features_l], dtype=np.float32)
        desc_r = np.array([f.descriptor_ for f in features_r], dtype=np.float32)

        print(f'Nb desc gauche {len(desc_l)} et droit {len(desc_r)}')
        
        knn_matches = self.matcher.knnMatch(desc_l, desc_r, k=2)
        t3 = time.time()
        print(f'Nb match {len(knn_matches)}')

        lowe = Params.get('lowe_ratio')
        pts_l_brut = []
        pts_r_brut = []
        idx_l_bruts_temp = []

        for m in knn_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * lowe:
                idx_l = m[0].queryIdx
                idx_r = m[0].trainIdx

                pts_l_brut.append(features_l[idx_l].position_.pt)
                pts_r_brut.append(features_r[idx_r].position_.pt)
                idx_l_bruts_temp.append(idx_l)

        print(f'Nb match {len(pts_l_brut)} post Lowe')
        if len(pts_l_brut) < 10:
            return False
        t4 =time.time()

        pts_l_brut = np.float32(pts_l_brut)
        pts_r_brut = np.float32(pts_r_brut)

        # Filtrage ransac matrice essentielle
        K1 = self.params_stero_['mtx1']
        K2 = self.params_stero_['mtx2']

        E, mask = cv.findEssentialMat(
            pts_l_brut, pts_r_brut,
            cameraMatrix=K1,
            method=cv.RANSAC, prob=0.999, threshold=Params.get('essential_ransac_threshold')
        )
        print(f'Nb match {len(pts_l_brut)} post Lowe')
        if mask is None:
            return False

        mask = mask.ravel() == 1
        
        pts_l_good = pts_l_brut[mask]
        pts_r_good = pts_r_brut[mask]

        idx_l_bruts_arr = np.array(idx_l_bruts_temp)
        idx_l_bruts = idx_l_bruts_arr[mask].tolist()
        t5 = time.time()
        print(f'Nb match {len(pts_l_good)} post Lowe+essential ransac')
        print(f'{t2-t1}s : loading desc')
        print(f'{t3-t2}s : matching desc')
        print(f'{t4-t3}s : test lowe')
        print(f'{t5-t4}s : ransac essential')

        # Triangulation
        
        P1_unrect = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2_unrect = K2 @ np.hstack((self.params_stero_['R'], self.params_stero_['T']))

        points4Dlocal = cv.triangulatePoints(P1_unrect, P2_unrect, pts_l_good.T, pts_r_good.T)
        points3Dlocal = (points4Dlocal[:3, :] / points4Dlocal[3, :]).T
        t6 =time.time()
        depth_min = Params.get('depth_min')
        depth_max = Params.get('depth_max')
        points_valides = 0
        for i, pt_local in enumerate(points3Dlocal):
            if depth_min < pt_local[2] < depth_max:
                idx_l = idx_l_bruts[i]
                feature_left = features_l[idx_l]
                # Filtre d'outliers
                # number_neighbors = 0
                # for _,autre_pt in enumerate(points3Dlocal):
                #     if np.linalg.norm(autre_pt-pt_local) < self.radius:
                #         number_neighbors += 1
                # if number_neighbors >= self.minimal_number_neigbors:
                new_mp = MapPoint.create_new_mappoint(position=pt_local)
                feature_left.map_point_ = new_mp
                new_mp.add_observation(feature_left)
                
                self.map_.insert_map_point(new_mp)
                points_valides += 1

        if points_valides >= 10:
            self.current_frame_.pose_ = np.eye(4) 
            self.current_frame_.set_keyframe()
            self.map_.insert_keyframes(self.current_frame_)
            self.last_keyframe_ = self.current_frame_

            self.status_ = FrontendStatus.TRACKING_GOOD
            print(f'Init OK - {points_valides} points insérés dans la Map')
            return True

        return False

    def tracking(self): 
        self.current_frame_.preprocess()
        self.current_frame_.extract_features()

        previous_feature3D = []
        previous_desc = []

        if self.previous_frame_ is not None: 
            for feature in self.previous_frame_.features_left_: 
                if feature.map_point_ is not None and not feature.is_outlier_:
                    previous_feature3D.append(feature)
                    previous_desc.append(feature.descriptor_)

        if len(previous_desc) < 15:
            self.status_ = FrontendStatus.LOST
            print("Tracking LOST : Pas assez de points 3D dans la frame précédente.")
            return False
        # for i in range(len(self.current_frame_.features_left_)):
        #     print(np.shape(self.current_frame_.features_left_[i]))
        previous_desc = np.array(previous_desc, dtype=np.float32)
        new_desc = np.array([f.descriptor_ for f in self.current_frame_.features_left_], dtype=np.float32)

        knn_matches = self.matcher.knnMatch(previous_desc, new_desc, k=2)

        matched_3d_pts = []
        matched_2d_pts = []
        good_matches_new_idx = []
        good_matches_old_idx = []

        lowe = Params.get('lowe_ratio')
        for match in knn_matches:
            if len(match) == 2 and match[0].distance < match[1].distance * lowe:
                m = match[0]
                old_feature = previous_feature3D[m.queryIdx]
                new_feature = self.current_frame_.features_left_[m.trainIdx]

                matched_3d_pts.append(old_feature.map_point_.pos_)
                matched_2d_pts.append(new_feature.position_.pt)

                good_matches_old_idx.append(m.queryIdx)
                good_matches_new_idx.append(m.trainIdx)

        if len(matched_3d_pts) >= 15:
            pts3d_arr = np.float32(matched_3d_pts)
            pts2d_arr = np.float32(matched_2d_pts)
            
            K1 = self.params_stero_['mtx1']
            
            # Image déjà propre -> distCoeffs = None
            success, rvec_new, tvec_new, inliers = cv.solvePnPRansac(
                pts3d_arr, pts2d_arr, K1, None,
                flags=cv.SOLVEPNP_EPNP,
                iterationsCount=100,
                reprojectionError=Params.get('pnp_reprojection_error')
            )
            
            if success and inliers is not None and len(inliers) >= 10:
                R_new, _ = cv.Rodrigues(rvec_new)
                T_cw_new = np.eye(4)
                T_cw_new[:3, :3] = R_new
                T_cw_new[:3, 3] = tvec_new.flatten()
                
                self.current_frame_.pose_ = T_cw_new
                self.num_frames_since_last_kf_ += 1
                
                for i in inliers.flatten():
                    idx_old = good_matches_old_idx[i]
                    idx_new = good_matches_new_idx[i]
                    
                    map_point = previous_feature3D[idx_old].map_point_
                    new_feature = self.current_frame_.features_left_[idx_new]
                    
                    new_feature.map_point_ = map_point
                    map_point.add_observation(new_feature)

                if self.need_new_keyframe(len(inliers), len(previous_feature3D)):
                    self.insert_keyframe()

                self.status_ = FrontendStatus.TRACKING_GOOD
                print(f"Tracking OK. Pose calculée avec {len(inliers)} inliers PnP.")
                return True
            else:
                self.status_ = FrontendStatus.TRACKING_BAD
                print("Tracking BAD. PnP a échoué.")
                return False
        else:
            self.status_ = FrontendStatus.LOST
            print("Tracking LOST. Pas assez de correspondances temporelles SIFT.")
            return False

    def reset(self):
        print("SYSTEM RESET - TRACKING IS LOST")
        self.status_ = FrontendStatus.INITING
        self.previous_frame_ = None
        return True

    def need_new_keyframe(self, num_inliers, num_previous_pts):
        #ratio_survie = num_inliers / max(1, num_previous_pts)

        if num_inliers < Params.get('kf_min_inliers'):
            return True

        if self.num_frames_since_last_kf_ > Params.get('kf_max_frames'):
            return True

        last_kf = self.map_.get_active_keyframes()[-1]
        translation = np.linalg.norm(self.current_frame_.pose_[:3, 3] - last_kf.pose_[:3, 3])
        R_rel = self.current_frame_.pose_[:3, :3] @ last_kf.pose_[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)) * 180 / np.pi

        if translation > Params.get('kf_translation') or angle > Params.get('kf_angle_deg'):
            return True

        return False
    
    def insert_keyframe(self):
        self.current_frame_.set_keyframe()
        self.map_.insert_keyframes(self.current_frame_)
        self.last_keyframe_ = self.current_frame_
        self.num_frames_since_last_kf_ = 0

        self.create_new_landmarks()

        if self.backend_ is not None:
            optim_thread = threading.Thread(target=self.backend_.update_map)
            optim_thread.daemon = True 
            optim_thread.start()

    def create_new_landmarks(self):
        features_l = self.current_frame_.features_left_
        features_r = self.current_frame_.features_right_

        idx_l_orphelins = [i for i, f in enumerate(features_l) if f.map_point_ is None]

        if not idx_l_orphelins or len(features_r) == 0:
            return
        desc_l = np.array([features_l[i].descriptor_ for i in idx_l_orphelins], dtype=np.float32)
        desc_r = np.array([f.descriptor_ for f in features_r], dtype=np.float32)

        knn_matches = self.matcher.knnMatch(desc_l, desc_r, k=2)

        lowe = Params.get('lowe_ratio')
        pts_l_brut = []
        pts_r_brut = []
        idx_l_bruts_temp = []

        for m in knn_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * lowe:
                idx_l_local = m[0].queryIdx
                idx_l_global = idx_l_orphelins[idx_l_local]
                idx_r = m[0].trainIdx

                pts_l_brut.append(features_l[idx_l_global].position_.pt)
                pts_r_brut.append(features_r[idx_r].position_.pt)
                idx_l_bruts_temp.append(idx_l_global)

        if len(pts_l_brut) < 10:
            return

        pts_l_brut = np.float32(pts_l_brut)
        pts_r_brut = np.float32(pts_r_brut)

        K1 = self.params_stero_['mtx1']
        K2 = self.params_stero_['mtx2']

        E, mask = cv.findEssentialMat(
            pts_l_brut, pts_r_brut,
            cameraMatrix=K1,
            method=cv.RANSAC, prob=0.999, threshold=Params.get('essential_ransac_threshold')
        )

        if mask is None:
            return

        mask = mask.ravel() == 1
        pts_l_good = pts_l_brut[mask]
        pts_r_good = pts_r_brut[mask]

        idx_l_bruts_arr = np.array(idx_l_bruts_temp)
        idx_l_bruts = idx_l_bruts_arr[mask].tolist()

        # Triangulation directe avec K1 et K2 car on a des pixels.
        P1_unrect = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2_unrect = K2 @ np.hstack((self.params_stero_['R'], self.params_stero_['T']))

        points4Dlocal = cv.triangulatePoints(P1_unrect, P2_unrect, pts_l_good.T, pts_r_good.T)
        points3Dlocal = (points4Dlocal[:3, :] / points4Dlocal[3, :]).T

        T_wc = np.linalg.inv(self.current_frame_.pose_)

        depth_min = Params.get('depth_min')
        depth_max = Params.get('depth_max')
        points_insere = 0
        for i, pt_local in enumerate(points3Dlocal):
            if depth_min < pt_local[2] < depth_max:
                pt_local_homo = np.append(pt_local, 1.0)
                pt_global = (T_wc @ pt_local_homo)[:3]

                new_mp = MapPoint.create_new_mappoint(position=pt_global)
                feat_l = features_l[idx_l_bruts[i]]
                
                feat_l.map_point_ = new_mp
                new_mp.add_observation(feat_l)
                
                self.map_.insert_map_point(new_mp)
                points_insere += 1
                
        print(f"[MAP] + {points_insere} nouveaux points projetés en global.")