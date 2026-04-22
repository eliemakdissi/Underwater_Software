# Implementation of the Frame class

import cv2 as cv
import numpy as np
import threading
import pickle
from Feature import Feature

class Frame:
    
    """ with open('/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/calibration/param/stereo_a_lenvers.pkl', 'rb') as f:
        params = pickle.load(f) """
    params = {'mtx1': np.array([[1.25051139e+03, 0.00000000e+00, 9.83089787e+02],
       [0.00000000e+00, 1.25043408e+03, 5.31404956e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), 'dist1': np.array([[-0.32465656,  0.19275159, -0.00075876,  0.0038447 , -0.0540854 ]]), 'mtx2': np.array([[1.24269700e+03, 0.00000000e+00, 9.67192665e+02],
       [0.00000000e+00, 1.24336262e+03, 5.71800196e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), 'dist2': np.array([[-0.32106801,  0.18508454,  0.00183416,  0.0011332 , -0.04652458]]), 'R': np.array([[ 0.99926041, -0.00330228,  0.03831103],
       [ 0.00421827,  0.99970658, -0.02385302],
       [-0.03822102,  0.02399699,  0.99898113]]), 'T': np.array([[-0.04065602],
       [-0.0002887 ],
       [-0.00069275]]), 'R1': np.array([[ 0.998469  ,  0.00420484,  0.05515408],
       [-0.00353962,  0.99991988, -0.01215338],
       [-0.05520077,  0.01193954,  0.99840389]]), 'R2': np.array([[ 0.99982966,  0.00709983,  0.01703628],
       [-0.00730462,  0.99990144,  0.01198936],
       [-0.01694948, -0.01211176,  0.99978299]]), 'P1': np.array([[1.19671761e+03, 0.00000000e+00, 9.14563248e+02, 0.00000000e+00],
       [0.00000000e+00, 1.19671761e+03, 5.57418053e+02, 0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]]), 'P2': np.array([[ 1.19671761e+03,  0.00000000e+00,  9.14563248e+02,
        -4.86620583e+01],
       [ 0.00000000e+00,  1.19671761e+03,  5.57418053e+02,
         0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
         0.00000000e+00]]), 'roi1': (0, 0, 1920, 1080), 'roi2': (0, 0, 1920, 1080), 'Q': np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -9.14563248e+02],
       [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        -5.57418053e+02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.19671761e+03],
       [ 0.00000000e+00,  0.00000000e+00,  2.45924166e+01,
        -0.00000000e+00]])}
    # Preprocessing
    '''
    mapl_x, mapl_y = cv.initUndistortRectifyMap(params['mtx1'], params['dist1'], params['R1'], params['P1'], (1920,1080), cv.CV_32FC1)
    mapr_x, mapr_y = cv.initUndistortRectifyMap(params['mtx2'], params['dist2'], params['R2'], params['P2'], (1920,1080), cv.CV_32FC1)
    '''
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Feature detection
    sift = cv.SIFT_create(contrastThreshold=0.05)
    orb = cv.ORB_create(nfeatures=10000, scaleFactor=1.2, nlevels=8)
    akaze = cv.AKAZE_create(threshold = 0.001, diffusivity = cv.KAZE_DIFF_CHARBONNIER)

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
        

        self.clean_left_img_ = Frame.clahe.apply(img_l_undist[:,:,1])
        self.clean_right_img_ = Frame.clahe.apply(img_r_undist[:,:,1])
        '''
        self.clean_left_img_ = img_l_undist[:,:,1]
        self.clean_right_img_ = img_r_undist[:,:,1]
        '''

        return True
    
    def extract_features(self):
        key_l, desc_l = Frame.sift.detectAndCompute(self.clean_left_img_, None)
        key_r, desc_r = Frame.sift.detectAndCompute(self.clean_right_img_, None)

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
        




            
        

