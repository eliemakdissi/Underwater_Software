# Implementation of the Frame class

import cv2 as cv
import numpy as np
import threading

class Frame:

    _next_id = 0

    def __init__(self, id=None, time_stamp=0.0, pose=None, left_img=None, right_img=None):
        # Data members
        self.id_ = id if id is not None else Frame._next_id
        self.keyframe_id_ = 0
        self.is_keyframe_ = False
        self.time_stamp_ = time_stamp
        
        # Pose 
        self.pose_ = pose if pose is not None else np.eye(4)
        
        # Thread safety
        self._pose_mutex = threading.Lock()
        
        # Images (OpenCV Mat en C++ devient numpy.ndarray en Python)
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
    def create_frame():
        new_frame = Frame(id=Frame._next_id)
        Frame._next_id += 1
        return new_frame
