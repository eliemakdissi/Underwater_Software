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
        
        self.map_ = map

        self.matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

    

    def add_frame (self, frame : Frame):
        self.current_frame_ = frame

        if self.status_ is FrontendStatus.INITING:
            self.stero_init()

        elif self.status_ is FrontendStatus.TRACKING_GOOD or FrontendStatus.TRACKING_BAD:
            self.tracking()

        elif self.status_ is FrontendStatus.LOST:
            self.reset()

        self.previous_frame_ = self.current_frame_

        return True


    def stero_init(self) : 
        self.current_frame_.preprocess()
        self.current_frame_.extract_features()
        self.current_frame_.compute_bins_r()

        features_l = self.current_frame_.features_left_
        features_r = self.current_frame_.features_right_

        # Matching avec bins
        bins_r = self.current_frame_.bins_r

        for i, feature in enumerate(features_l):

            central_bin = (features_l[i].position_[1])//self.current_frame_.BIN_SIZE
            bins_to_research = [central_bin+i for i in range(-1,2,1) if central_bin+i in bins_r.keys()]

            idx_r_to_research = []
            for b in bins_to_research:
                idx_r_to_research.extend(bins_r[b])

            if len(idx_r_to_research) < 2:
                continue

            desc_l_unique = np.array([feature.descriptor_], dtype=np.float32)
            desc_r_research = np.array([features_r[i].descriptor_ for i in idx_r_to_research], dtype=np.float32)

            idx_l_bruts = []
            idx_r_bruts = []

            knn_matchs = self.matcher.knnMatch(desc_l_unique, desc_r_research)
            lowe = 0.75

            # Test de Lowe
            if len(knn_matchs)>0 and len(knn_matchs[0])==2:
                    m, n = knn_matchs[0]
                    if m <= n*lowe:
                        
                        global_r_index = idx_r_to_research[m.trainIdx]

                        idx_l_bruts.append(i)
                        idx_r_bruts.append(global_r_index)

        pts_l = np.array(features_l[idx_l_bruts])
        pts_r = np.array(features_r[idx_r_bruts])

        # Triangulation
        points4D = cv.triangulatePoints(self.params_stero_['P1'], self.params_stero_['P2'],pts_l.T, pts_r.T)
        points3D = (points4D[:3, :] / points4D[3, :]).T


        points_valides = 0

        for i, point in enumerate(points3D):
            if 0.1 < point[2] < 15.0:
                idx_l = idx_l_bruts[i]
                feature_left = features_l[i]

                # On crée le nouveau point 3D
                new_mappoint = MapPoint.create_new_mappoint(position=point)

                # On déclare au feature sa position en 3D
                feature_left.map_points_ = new_mappoint

                # On déclare au point 3D qu'il est vu par ce feature dans l'image gauche
                new_mappoint.add_observation(feature_left)

                # On insère le point 3D dans la map globale
                self.map_.insert_map_point(new_mappoint)

                points_valides +=1
                
        if points_valides >=10 :

            self.current_frame_.set_keyframe()
            self.map_.insert_keyframes(self.current_frame_)
            self.status_ = FrontendStatus.TRACKING_GOOD
            

    def tracking(self) : pass

    def reset(self) :

        print("SYSTEM RESET - TRACKING IS LOST")
        self.status_ = FrontendStatus.INITING
        self.previous_frame_ = None
        # Vide-t-on la map entièrement ici ou est-ce qu'on la garde encore pour refaire du Loop Closure ?
        return True
