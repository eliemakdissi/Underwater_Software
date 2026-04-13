import threading
import cv2 as cv
import numpy as np

from Frame import Frame
from Map import Map
from MapPoint import MapPoint

class FrontendStatus():
    INITING = 0
    TRACKING_GOOD = 1
    TRACKING_BAD = 2
    LOST = 3

class Frontend():

    def __init__(self, params_stereo_, map : Map):

        self.status_ = FrontendStatus.INITING
        self.params_stero_ = params_stereo_

        self.current_frame_ = None
        self.previous_frame_ = None
        self.last_keyframe_ = None
        self.num_frames_since_last_kf_ = 0

        self.map_ = map

        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

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


    def stero_init(self) : 
        self.current_frame_.preprocess()
        self.current_frame_.extract_features()
        self.current_frame_.compute_bins()

        features_l = self.current_frame_.features_left_
        features_r = self.current_frame_.features_right_

        idx_l_bruts = []
        idx_r_bruts = []

        # Matching avec bins
        bins_l = self.current_frame_.bins_l
        bins_r = self.current_frame_.bins_r

        for bin_idx, idx_l_research in bins_l.items():
            bins_to_research=[bin_idx + i for i in range(-1,2) if bin_idx+i in bins_r.keys()]
            idx_r_research = []
            for b in bins_to_research:
                idx_r_research.extend(bins_r[b])

            desc_l = np.array([features_l[i].descriptor_ for i in idx_l_research], dtype=np.uint8)
            desc_r = np.array([features_r[i].descriptor_ for i in idx_r_research], dtype=np.uint8)

            knnMatches = self.matcher.knnMatch(desc_l, desc_r, k=2)

            # Test de Lowe
            LOWE =  0.75
            for match in knnMatches:
                if len(match)==2:
                    m, n = match
                    if m.distance <= n.distance * LOWE : 

                        idx_l_bruts.append(idx_l_research[m.queryIdx])
                        idx_r_bruts.append(idx_r_research[m.trainIdx])


        # Matching avec bins mais un indice à la fois
        '''
        for i, feature in enumerate(features_l):

            central_bin = (features_l[i].position_.pt[1])//self.current_frame_.BIN_SIZE
            bins_to_research = [central_bin+i for i in range(-1,2,1) if central_bin+i in bins_r.keys()]

            idx_r_to_research = []
            for b in bins_to_research:
                idx_r_to_research.extend(bins_r[b])

            if len(idx_r_to_research) < 2:
                continue

            desc_l_unique = np.array([feature.descriptor_], dtype=np.float32)
            desc_r_research = np.array([features_r[i].descriptor_ for i in idx_r_to_research], dtype=np.float32)

            knn_matchs = self.matcher.knnMatch(desc_l_unique, desc_r_research, k=2)
            lowe = 0.75

            # Test de Lowe
            if len(knn_matchs)>0 and len(knn_matchs[0])==2:
                    m, n = knn_matchs[0]
                    if m.distance <= n.distance*lowe:
                        
                        global_r_index = idx_r_to_research[m.trainIdx]

                        idx_l_bruts.append(i)
                        idx_r_bruts.append(global_r_index)
        '''

        pts_l = np.float32([features_l[i].position_.pt for i in idx_l_bruts])
        pts_r = np.float32([features_r[i].position_.pt for i in idx_r_bruts])

        # Triangulation
        points4D = cv.triangulatePoints(self.params_stero_['P1'], self.params_stero_['P2'],pts_l.T, pts_r.T)
        points3D = (points4D[:3, :] / points4D[3, :]).T


        points_valides = 0

        for i, point in enumerate(points3D):
            if 0.1 < point[2] < 15.0:
                idx_l = idx_l_bruts[i]
                feature_left = features_l[idx_l]

                # Nouveau pt 3D
                new_mappoint = MapPoint.create_new_mappoint(position=point)

                # feature <-> pt 3D
                feature_left.map_point_ = new_mappoint

                # pt 3D est vu par ce feature
                new_mappoint.add_observation(feature_left)

                # pt 3d -> map globale
                self.map_.insert_map_point(new_mappoint)

                points_valides +=1
                
        if points_valides >=10 :

            self.current_frame_.pose_ = np.eye(4) # On définit l'origine de notre
            self.current_frame_.set_keyframe()
            self.map_.insert_keyframes(self.current_frame_)
            self.last_keyframe_ = self.current_frame_

            self.status_ = FrontendStatus.TRACKING_GOOD
            print(f'Init OK - {points_valides} insered in the Map')
            return True


    def tracking(self) : 

        self.current_frame_.preprocess()
        self.current_frame_.extract_features()

        previous_feature3D = []
        previous_desc = []

        if self.previous_frame_ is not None : 
            for feature in self.previous_frame_.features_left_ : 
                if feature.map_point_ is not None and not feature.is_outlier_ :
                    previous_feature3D.append(feature)
                    previous_desc.append(feature.descriptor_)

        if len(previous_desc) < 15:
            self.status_ = FrontendStatus.LOST
            print("Tracking LOST : not enough 3D points in the previous frame")
            return False
        
        previous_desc = np.array(previous_desc, dtype=np.uint8)
        new_desc = np.array([feature.descriptor_ for feature in self.current_frame_.features_left_], dtype=np.uint8)

        knnMatchs = self.matcher.knnMatch(previous_desc, new_desc, k=2)

        matched_3d_pts = []
        matched_2d_pts = []
        good_matches_new_idx = []
        good_matches_old_idx = []

        lowe = 0.75
        for match in knnMatchs:
            if len(match)==2:
                m,n = match
                if m.distance < lowe * n.distance :
                    
                    old_feature = previous_feature3D[m.queryIdx]
                    new_feature = self.current_frame_.features_left_[m.trainIdx]

                    matched_3d_pts.append(old_feature.map_point_.pos_)
                    matched_2d_pts.append(new_feature.position_.pt)

                    good_matches_old_idx.append(m.queryIdx)
                    good_matches_new_idx.append(m.trainIdx)


        if len(matched_3d_pts) >= 15:
            pts3d_arr = np.float32(matched_3d_pts)
            pts2d_arr = np.float32(matched_2d_pts)
            K = self.params_stero_['P1'][:3, :3]
            
            # PnP + Ransac
            success, rvec_new, tvec_new, inliers = cv.solvePnPRansac(
                pts3d_arr, pts2d_arr, K, None, 
                flags=cv.SOLVEPNP_ITERATIVE,
                iterationsCount=100,
                reprojectionError=10.0 # Tolérance de 3 pixels
            )
            
            if success and inliers is not None and len(inliers) >= 10:
                # Mise à jour de la pose
                R_new, _ = cv.Rodrigues(rvec_new)
                T_cw_new = np.eye(4)
                T_cw_new[:3, :3] = R_new
                T_cw_new[:3, 3] = tvec_new.flatten()
                
                self.current_frame_.pose_ = T_cw_new
                self.num_frames_since_last_kf_ += 1
                
                # On lie les nouvelles features aux anciens pts 3D
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
                print("Tracking BAD. PnP a échoué (mouvement trop brusque ou faux matchs).")
                return False
        else:
            self.status_ = FrontendStatus.LOST
            print("Tracking LOST. Pas assez de correspondances SIFT.")
            return False
        

    def reset(self) :

        print("SYSTEM RESET - TRACKING IS LOST")
        self.status_ = FrontendStatus.INITING
        self.previous_frame_ = None
        # Vide-t-on la map entièrement ici ou est-ce qu'on la garde encore pour refaire du Loop Closure ?
        return True


    def need_new_keyframe(self, num_inliers, num_previous_pts):

        ratio_survie = num_inliers / max(1, num_previous_pts)

        if ratio_survie < 0.6 or num_inliers < 50: 
            print(f"[KF] Création : ratio de survie faible ({ratio_survie*100:.1f}%)")
            return True
        
        if self.num_frames_since_last_kf_ > 15 :
            return True
        
        last_kf = list(self.map_.get_active_keyframes().values())[-1] 
        
        translation = np.linalg.norm(
            self.current_frame_.pose_[:3, 3] - last_kf.pose_[:3, 3]
        )
        
        # Rotation angle
        R_rel = self.current_frame_.pose_[:3, :3] @ last_kf.pose_[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)) * 180 / np.pi
        
        if translation > 0.15 or angle > 8.0:  # 15cm ou 8°
            print(f"[KF] Critère géométrique: {translation:.2f}m, {angle:.1f}°")
            return True
        
        return False
    
    def insert_keyframe(self):

        self.current_frame_.set_keyframe()
        self.map_.insert_keyframes(self.current_frame_)
        self.last_keyframe_ = self.current_frame_
        self.num_frames_since_last_kf_ = 0

        self.create_new_landmarks()

        # Backend ici pour démarrer l'optim

    def create_new_landmarks(self):

        features_l = self.current_frame_.features_left_
        features_r = self.current_frame_.features_right_
        
        # On s'assure que les bins sont calculés pour la frame courante
        self.current_frame_.compute_bins() 

        # On récupère les bins globaux déjà calculés
        bins_l = self.current_frame_.bins_l
        bins_r = self.current_frame_.bins_r

        idx_l_bruts = []
        idx_r_bruts = []

        for bin_idx, idx_l_all in bins_l.items():
            
            # --- LE FILTRE CRUCIAL ---
            # Dans ce bin, on ne garde QUE les indices des features qui n'ont PAS de map_point
            idx_l_research = [i for i in idx_l_all if features_l[i].map_point_ is None]
            
            # S'il n'y a aucun point orphelin dans ce bin, on passe au suivant
            if not idx_l_research:
                continue

            bins_to_research = [bin_idx + i for i in range(-1, 2) if (bin_idx + i) in bins_r.keys()]
            idx_r_research = []
            for b in bins_to_research:
                idx_r_research.extend(bins_r[b])

            if len(idx_r_research) < 2:
                continue

            # On utilise maintenant idx_l_research (les orphelins) pour le desc_l
            desc_l = np.array([features_l[i].descriptor_ for i in idx_l_research], dtype=np.uint8)
            desc_r = np.array([features_r[i].descriptor_ for i in idx_r_research], dtype=np.uint8)

            knnMatches = self.matcher.knnMatch(desc_l, desc_r, k=2)

            LOWE = 0.75
            for match in knnMatches:
                if len(match) == 2:
                    m, n = match
                    if m.distance <= n.distance * LOWE:
                        idx_l_bruts.append(idx_l_research[m.queryIdx])
                        idx_r_bruts.append(idx_r_research[m.trainIdx])

        pts_l = np.float32([features_l[i].position_.pt for i in idx_l_bruts])
        pts_r = np.float32([features_r[i].position_.pt for i in idx_r_bruts])

        # Triangulation
        points4Dlocal = cv.triangulatePoints(self.params_stero_['P1'], self.params_stero_['P2'],pts_l.T, pts_r.T)
        points3Dlocal = (points4Dlocal[:3, :] / points4Dlocal[3, :]).T


        T_wc = np.linalg.inv(self.current_frame_.pose_)
        
        points_insere = 0
        for i, pt_local in enumerate(points3Dlocal):
            if 0.1 < pt_local[2] < 15.0:
                # Ajout du '1' pour multiplier avec la matrice 4x4
                pt_local_homo = np.append(pt_local, 1.0)
                # Multiplication pour passer du local au global
                pt_global = (T_wc @ pt_local_homo)[:3]

                new_mp = MapPoint.create_new_mappoint(position=pt_global)
                feat_l = features_l[idx_l_bruts[i]]
                
                feat_l.map_point_ = new_mp
                new_mp.add_observation(feat_l)
                
                self.map_.insert_map_point(new_mp)
                points_insere += 1
                
        print(f"[MAP] + {points_insere} nouveaux points projetés en global.")