# Implementation of the Frame class

import cv2 as cv
import numpy as np
import threading
import pickle
import Params
from Feature import Feature

class Frame:

    with open('SLAM/calibration/param/stereo_a_lenvers.pkl', 'rb') as f:
        params = pickle.load(f)

    # Preprocessing
    '''
    mapl_x, mapl_y = cv.initUndistortRectifyMap(params['mtx1'], params['dist1'], params['R1'], params['P1'], (1920,1080), cv.CV_32FC1)
    mapr_x, mapr_y = cv.initUndistortRectifyMap(params['mtx2'], params['dist2'], params['R2'], params['P2'], (1920,1080), cv.CV_32FC1)
    '''
    orb = cv.ORB_create(nfeatures=10000, scaleFactor=1.2, nlevels=8)
    akaze = cv.AKAZE_create(threshold = 0.001, diffusivity = cv.KAZE_DIFF_CHARBONNIER)

    _sift = None
    _sift_version = -1
    _clahe = None
    _clahe_version = -1

    @classmethod
    def get_sift(cls):
        v = Params.version('sift_contrast_threshold')
        if cls._sift is None or cls._sift_version != v:
            cls._sift = cv.SIFT_create(contrastThreshold=Params.get('sift_contrast_threshold'))
            cls._sift_version = v
        return cls._sift

    @classmethod
    def get_clahe(cls):
        v = Params.version('clahe_clip_limit')
        if cls._clahe is None or cls._clahe_version != v:
            cls._clahe = cv.createCLAHE(clipLimit=Params.get('clahe_clip_limit'), tileGridSize=(8, 8))
            cls._clahe_version = v
        return cls._clahe

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

        img_l_undist = cv.undistort(self.left_img_, Frame.params['mtx1'], Frame.params['dist1'], Frame.params['mtx1'])
        img_r_undist = cv.undistort(self.right_img_, Frame.params['mtx2'], Frame.params['dist2'], Frame.params['mtx2'])


        clahe = Frame.get_clahe()
        self.clean_left_img_ = clahe.apply(img_l_undist[:,:,1])
        self.clean_right_img_ = clahe.apply(img_r_undist[:,:,1])

        self.clean_left_img_ = img_l_undist[:,:,1]
        self.clean_right_img_ = img_r_undist[:,:,1]

        self.color_left_img_ = img_l_undist
        self.color_right_img_ = img_r_undist

        return True
    
    def extract_features(self):
        sift = Frame.get_sift()
        key_l, desc_l = sift.detectAndCompute(self.clean_left_img_, None)
        key_r, desc_r = sift.detectAndCompute(self.clean_right_img_, None)

        if desc_l is not None:
            desc_l = np.atleast_2d(desc_l)
            for i in range(len(key_l)):
                new_feature = Feature(frame=self, keypoint=key_l[i], descriptor=desc_l[i].copy())
                new_feature.is_on_left_image_ = True
                new_feature.color_ = Frame._sample_color(self.color_left_img_, key_l[i].pt)
                self.features_left_.append(new_feature)

        if desc_r is not None:
            desc_r = np.atleast_2d(desc_r)
            for i in range(len(key_r)):
                new_feature = Feature(frame=self, keypoint=key_r[i], descriptor=desc_r[i].copy())
                new_feature.is_on_left_image_ = False
                new_feature.color_ = Frame._sample_color(self.color_right_img_, key_r[i].pt)
                self.features_right_.append(new_feature)

        return True

    @staticmethod
    def _sample_color(bgr_img, pt):
        # cv.imread gives BGR; return (R, G, B) ints for plotting convenience.
        h, w = bgr_img.shape[:2]
        x = min(max(int(round(pt[0])), 0), w - 1)
        y = min(max(int(round(pt[1])), 0), h - 1)
        b, g, r = bgr_img[y, x]
        return (int(r), int(g), int(b))

    def compute_bins(self):

        self.bins_l = {}
        self.bins_r = {}
        self.BIN_SIZE = 25

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
        




            
        

