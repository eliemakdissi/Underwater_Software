# Implementation of the Frame class

import cv2 as cv
import numpy as np
import threading
import pickle
from Feature import Feature

class Frame:

    with open('SLAM/calibration/param/stereo_params_complets.pkl', 'rb') as f:
        params = pickle.load(f)

    # Preprocessing
    
    mapl_x, mapl_y = cv.initUndistortRectifyMap(params['mtx1'], params['dist1'], params['R1'], params['P1'], (1920,1080), cv.CV_32FC1)
    mapr_x, mapr_y = cv.initUndistortRectifyMap(params['mtx2'], params['dist2'], params['R2'], params['P2'], (1920,1080), cv.CV_32FC1)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Feature detection

    sift = cv.SIFT_create(contrastThreshold=0.02, edgeThreshold=10, nOctaveLayers=4)
    orb = cv.ORB_create(nfeatures=10000, scaleFactor=1.2, nlevels=8)


    _next_id = 0

    def __init__(self, id=None, time_stamp=0.0, pose=None, left_img=None, right_img=None):
        # Data members
        # A checker si on envoit que les paths ou directement les cv.imread
        self.id_ = id if id is not None else Frame._next_id
        self.keyframe_id_ = 0
        self.is_keyframe_ = False
        self.time_stamp_ = time_stamp
        
        # Pose 
        self.pose_ = pose if pose is not None else np.eye(4)
        
        # Thread safety
        self._pose_mutex = threading.Lock()
        
        # Images 
        self.left_img_ = left_img
        self.right_img_ = right_img

        # Features lists
        self.features_left_ = []
        self.features_right_ = []

    @property
    def pose(self):
        """ Getter thread-safe pour la pose """
        with self._pose_mutex:
            return self.pose_

    @pose.setter
    def pose(self, new_pose):
        """ Setter thread-safe pour la pose """
        with self._pose_mutex:
            self.pose_ = new_pose

    def set_keyframe(self):
        self.is_keyframe_ = True
        self.keyframe_id_ = self.id_ 

    @staticmethod
    def create_frame(time_stamp : float, left_img, right_img):
        new_frame = Frame(id=Frame._next_id, time_stamp=time_stamp, left_img=left_img, right_img=right_img)
        Frame._next_id += 1
        return new_frame
    
    def preprocess(self): 

        img_l_rect = cv.remap(self.left_img_, Frame.mapl_x, Frame.mapl_y, cv.INTER_LINEAR)
        img_r_rect = cv.remap(self.right_img_, Frame.mapr_x, Frame.mapr_y, cv.INTER_LINEAR)

        self.clean_left_img_ = Frame.clahe.apply(img_l_rect[:,:,1])
        self.clean_right_img_ = Frame.clahe.apply(img_r_rect[:,:,1])

        return True
    
    def extract_features(self):
        key_l, desc_l = Frame.orb.detectAndCompute(self.clean_left_img_, None)
        key_r, desc_r = Frame.orb.detectAndCompute(self.clean_right_img_, None)

        for i in range(len(key_l)):
            new_feature = Feature(frame=self, keypoint=key_l[i], descriptor=desc_l[i])
            new_feature.is_on_left_image_= True
            self.features_left_.append(new_feature)
        for i in range(len(key_r)):
            new_feature = Feature(frame=self, keypoint=key_r[i], descriptor= desc_r[i])
            new_feature.is_on_left_image_= False
            self.features_right_.append(new_feature)
        return True

    def compute_bins(self):

        self.bins_l = {}
        self.bins_r = {}
        self.BIN_SIZE = 50

        for i, feature in enumerate(self.features_left_) :
            bin_nb = int(feature.position_.pt[1]/self.BIN_SIZE)

            if bin_nb not in self.bins_l:
                self.bins_l[bin_nb] = [i]
            else:
                self.bins_l[bin_nb].append(i)

        for i, feature in enumerate(self.features_right_) :
            bin_nb = int(feature.position_.pt[1]/self.BIN_SIZE)

            if bin_nb not in self.bins_r:
                self.bins_r[bin_nb] = [i]
            else:
                self.bins_r[bin_nb].append(i)

        return True
        




            
        

