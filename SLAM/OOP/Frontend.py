import threading
import cv2 as cv
import numpy as np

from Frame import Frame
from Map import Map
from MapPoint import MapPoint
from Backend import Backend
from database import LoopClosureDatabase
from vocabulary_tree import VocabularyTree


VOCABSIZE = 50

def verify_loop_geometrically(current_kf, candidate_kf, min_common_words=15):
        current_words = set(current_kf["words"])
        candidate_words = set(candidate_kf["words"])
        common = len(current_words.intersection(candidate_words))
        return common >= min_common_words

class FrontendStatus():
    INITING = 0
    TRACKING_GOOD = 1
    TRACKING_BAD = 2
    LOST = 3

class Frontend():

    def __init__(self, params_stereo_, map : Map, backend : Backend):

        self.status_ = FrontendStatus.INITING
        self.params_stero_ = params_stereo_

        self.current_frame_ = None
        self.previous_frame_ = None
        self.last_keyframe_ = None
        self.num_frames_since_last_kf_ = 0

        self.map_ = map

        self.matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

        self.backend_ = backend
        self.vocab_size = VOCABSIZE
        
        self.loop_db = LoopClosureDatabase(num_words=self.vocab_size)  # vocab_size à définir
        self.vocab_tree = VocabularyTree.load("mon_vocabulaire_sift.pkl")  # Charger le vocabulaire entraîné
        self.last_kf_desc = None  # Derniers descripteurs de keyframe

    

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
            LOWE = 0.95
            MAX_EPIPOLAR = 15

            for match in knnMatches:
                if len(match) == 2:
                    m, n = match
                    if m.distance <= n.distance * LOWE:
                        idx_l = idx_l_research[m.queryIdx]
                        idx_r = idx_r_research[m.trainIdx]

                        if np.abs(features_l[idx_l].position_.pt[1]-features_r[idx_r].position_.pt[1]) <= MAX_EPIPOLAR:
                            idx_l_bruts.append(idx_l)
                            idx_r_bruts.append(idx_r)
        """
        debug_matches = []
        for i in range(len(idx_l_bruts)):
            debug_matches.append(cv.DMatch(idx_l_bruts[i], idx_r_bruts[i], 0))

        kp_l = [f.position_ for f in features_l]
        kp_r = [f.position_ for f in features_r]

        img_stereo = cv.drawMatches(
            self.current_frame_.clean_left_img_, kp_l,
            self.current_frame_.clean_right_img_, kp_r,
            debug_matches, None, 
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0)
        )
            
        img_stereo_resized = cv.resize(img_stereo, (0, 0), fx=0.5, fy=0.5)
        cv.imshow("DEBUG : Stéréo Matching", img_stereo_resized)
        
        # 4. LA PAUSE : Le programme s'arrête ici jusqu'à ce que tu appuies sur une touche
        print(f"PAUSE DEBUG : {len(idx_l_bruts)} matchs trouvés. Appuie sur n'importe quelle touche pour continuer...")
        print(f'{len(self.current_frame_.features_left_)} feature left et {len(self.current_frame_.features_right_)}')
        cv.waitKey(0)
        """


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
            print("MAPKEYFRAMES")
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
        
        previous_desc = np.array(previous_desc, dtype=np.float32)
        new_desc = np.array([feature.descriptor_ for feature in self.current_frame_.features_left_], dtype=np.float32)

        knnMatchs = self.matcher.knnMatch(previous_desc, new_desc, k=2)

        matched_3d_pts = []
        matched_2d_pts = []
        good_matches_new_idx = []
        good_matches_old_idx = []

        lowe = 0.95
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

        # Debug
        '''
        
        kp_l = []
        kp_r = []
        debug_matches = []

        for i in range(len(good_matches_new_idx)):
            # Attention : l'ancien index correspond à la liste des points 3D filtrés
            old_idx = good_matches_old_idx[i]
            old_feature = previous_feature3D[old_idx] 
            
            # Le nouvel index correspond à toutes les features de la frame actuelle
            new_idx = good_matches_new_idx[i]
            new_feature = self.current_frame_.features_left_[new_idx]

            kp_l.append(old_feature.position_)
            kp_r.append(new_feature.position_)
            
            # Puisque nos listes kp_l et kp_r sont parfaitement alignées, 
            # on relie simplement la position i à la position i
            debug_matches.append(cv.DMatch(i, i, 0))

        img_stereo = cv.drawMatches(
            self.previous_frame_.clean_left_img_, kp_l,
            self.current_frame_.clean_left_img_, kp_r,
            debug_matches, None, 
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0)
        )
            
        img_stereo_resized = cv.resize(img_stereo, (0, 0), fx=0.5, fy=0.5)
        cv.imshow("DEBUG : Temporal Tracking", img_stereo_resized)
        
        print(f"⏸PAUSE DEBUG : {len(good_matches_new_idx)} matchs temporels trouvés. Appuie sur une touche...")
        cv.waitKey(0)

        '''

        if len(matched_3d_pts) >= 15:
            pts3d_arr = np.float32(matched_3d_pts)
            pts2d_arr = np.float32(matched_2d_pts)
            K = self.params_stero_['P1'][:3, :3]
            
            # PnP + Ransac
            success, rvec_new, tvec_new, inliers = cv.solvePnPRansac(
                pts3d_arr, pts2d_arr, K, None, 
                flags=cv.SOLVEPNP_EPNP,
                iterationsCount=100,
                reprojectionError=15.0
            )
            
            if success and inliers is not None and len(inliers) >= 10:
                # Mise à jour de la pose
                R_new, _ = cv.Rodrigues(rvec_new)
                T_cw_new = np.eye(4)
                T_cw_new[:3, :3] = R_new
                T_cw_new[:3, 3] = tvec_new.flatten()

                print(R_new, tvec_new.flatten())
                
                self.current_frame_.pose_ = T_cw_new
                self.num_frames_since_last_kf_ += 1


                # Debug
                '''
                kp_l = []
                kp_r = []
                debug_matches = []

                # inliers est un tableau 2D (ex: [[0], [3], [4]...]), on l'aplatit
                inliers_flat = inliers.flatten()

                for draw_idx, original_match_idx in enumerate(inliers_flat):
                    # On récupère les indices originaux validés par RANSAC
                    old_idx = good_matches_old_idx[original_match_idx]
                    new_idx = good_matches_new_idx[original_match_idx]
                    
                    old_feature = previous_feature3D[old_idx]
                    new_feature = self.current_frame_.features_left_[new_idx]

                    # On lie la nouvelle feature au point 3D (Ton code original)
                    map_point = old_feature.map_point_
                    new_feature.map_point_ = map_point
                    map_point.add_observation(new_feature)

                    # On prépare l'affichage
                    kp_l.append(old_feature.position_)
                    kp_r.append(new_feature.position_)
                    debug_matches.append(cv.DMatch(draw_idx, draw_idx, 0))

                # Dessin
                img_inliers = cv.drawMatches(
                    self.previous_frame_.clean_left_img_, kp_l,
                    self.current_frame_.clean_left_img_, kp_r,
                    debug_matches, None, 
                    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                    matchColor=(0, 255, 0)
                )
                    
                img_inliers_resized = cv.resize(img_inliers, (0, 0), fx=0.5, fy=0.5)
                cv.imshow("DEBUG : Inliers RANSAC", img_inliers_resized)
                
                print(f"⏸️ PAUSE DEBUG : {len(inliers_flat)} INLIERS validés. Appuie sur une touche...")
                cv.waitKey(0)
                '''
                
                # On lie les nouvelles features aux anciens pts 3D
                for i in inliers.flatten():
                    idx_old = good_matches_old_idx[i]
                    idx_new = good_matches_new_idx[i]
                    
                    map_point = previous_feature3D[idx_old].map_point_
                    new_feature = self.current_frame_.features_left_[idx_new]
                    
                    new_feature.map_point_ = map_point
                    map_point.add_observation(new_feature)


                if self.need_new_keyframe(len(inliers), len(previous_feature3D)):
                    print("KEYFRAMES ")
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
        
    def detect_loop_candidates(self, current_kf):
        """Détecte les boucles candidates pour le keyframe courant."""
        # 1. Extraire les mots visuels pour le keyframe courant
        words_in_image = self.vocab_tree.transform(current_kf.desc)
        print("CANDIDATTTTTT")
        # 2. Rechercher les candidats dans la base
        candidates = self.loop_db.find_loop_candidates(
            words_in_image,
            current_kf_id=current_kf.id_,
            min_temporal_distance=10,  # Éviter les voisins temporels
            top_k=3
        )

        # 3. Vérification géométrique 
        verified_loops = []
        for candidate_id, score in candidates:
            candidate_kf = self.loop_db.get_keyframe(candidate_id)
            if verify_loop_geometrically(current_kf, candidate_kf):
                verified_loops.append((candidate_id, score))
        for candidate_id, score in candidates:
            candidate_kf = self.loop_db.get_keyframe(candidate_id)
            if verify_loop_geometrically(current_kf, candidate_kf):
                print(f"[LOOP] Boucle détectée avec KF_{candidate_id} (score={score:.3f})")
                self.show_loop_match(current_kf, candidate_kf)  
                verified_loops.append((candidate_id, score))
        return verified_loops

   
    

    
    def reset(self) :

        print("SYSTEM RESET - TRACKING IS LOST")
        self.status_ = FrontendStatus.INITING
        self.previous_frame_ = None
        # Vide-t-on la map entièrement ici ou est-ce qu'on la garde encore pour refaire du Loop Closure ?
        return True


    def need_new_keyframe(self, num_inliers, num_previous_pts):

        ratio_survie = num_inliers / max(1, num_previous_pts)

        if ratio_survie < 0.6 or num_inliers < 50: 
            print(f"new keyfram : ratio de survie ({ratio_survie*100:.1f}%)")
            return True
        
        if self.num_frames_since_last_kf_ > 15 :
            return True
        
        last_kf = self.map_.get_active_keyframes()[-1] 
        
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

        if self.backend_ is not None:

            optim_thread = threading.Thread(target=self.backend_.update_map)
            optim_thread.daemon = True 
            optim_thread.start()
        self.current_frame_.set_keyframe()
        self.map_.insert_keyframes(self.current_frame_)

        # Loop CLOSURE
        kf_data = {
            "id": self.current_frame_.id_,
            "desc": np.array([f.descriptor_ for f in self.current_frame_.features_left_ if f.map_point_ is not None]),
            "words": self.vocab_tree.transform([f.descriptor_ for f in self.current_frame_.features_left_ if f.map_point_ is not None]),
            "pose": self.current_frame_.pose_
        }
        self.loop_db.add_keyframe(kf_data)

        # Détecter les boucles candidates
        print("passage dans detect loop")
        loops = self.detect_loop_candidates(kf_data)
        if loops:
            print(f"[LOOP] Boucles détectées : {loops}")
            self.backend_.add_loop_constraints(loops)  # Transmettre au Backend

  
    def show_loop_match(self, kf1, kf2):
        """Affiche les deux keyframes candidates pour validation visuelle."""
        # Récupérer les images à partir des chemins (à adapter selon ta structure)
        img1 = cv.imread(kf1["path_L"])  # Supposons que tu stockes le path dans kf1/kf2
        img2 = cv.imread(kf2["path_L"])

        if img1 is None or img2 is None:
            print("Erreur : Impossible de charger les images.")
            return

        # Redimensionner pour un affichage côté à côté
        h = min(img1.shape[0], img2.shape[0])
        img1 = cv.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
        img2 = cv.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))

        # Concatenation horizontale
        combined = cv.hconcat([img1, img2])
        cv.putText(combined, f"Loop Candidate: KF_{kf1['id']} <-> KF_{kf2['id']}",
                    (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Affichage
        cv.imshow("Loop Closure Candidate", combined)
        cv.waitKey(1000)  # Attendre 1 seconde
        cv.destroyAllWindows()


    def create_new_landmarks(self):

        features_l = self.current_frame_.features_left_
        features_r = self.current_frame_.features_right_
        
        self.current_frame_.compute_bins() 

        # On récupère les bins globaux déjà calculés
        bins_l = self.current_frame_.bins_l
        bins_r = self.current_frame_.bins_r

        idx_l_bruts = []
        idx_r_bruts = []

        for bin_idx, idx_l_all in bins_l.items():
            
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

            desc_l = np.array([features_l[i].descriptor_ for i in idx_l_research], dtype=np.float32)
            desc_r = np.array([features_r[i].descriptor_ for i in idx_r_research], dtype=np.float32)

            knnMatches = self.matcher.knnMatch(desc_l, desc_r, k=2)

            LOWE = 0.95
            MAX_EPIPOLAR = 15
            for match in knnMatches:
                if len(match) == 2:
                    m, n = match
                    if m.distance <= n.distance * LOWE:
                        idx_l = idx_l_research[m.queryIdx]
                        idx_r = idx_r_research[m.trainIdx]

                        if np.abs(features_l[idx_l].position_.pt[1]-features_r[idx_r].position_.pt[1]) <= MAX_EPIPOLAR:
                            idx_l_bruts.append(idx_l)
                            idx_r_bruts.append(idx_r)

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