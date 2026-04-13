# Implementation of the Map class
import threading
import numpy as np

class Map:
    
    def __init__(self):

        self.landmarks_ = {}
        self.active_landmarks_ = {}
        self.keyframes_ = {}
        self.active_keyframes_ = {}

        self.num_active_keyframes_ = 7
        self.current_frame_ = None

        self.data_mutex_ = threading.Lock()

    def insert_keyframes(self, frame):

        with self.data_mutex_:
            self.current_frame_ = frame
            self.keyframes_[frame.id_] = frame
            self.active_keyframes_[frame.id_] = frame

        if len(self.active_keyframes_) > self.num_active_keyframes_:
            self.remove_old_keyframe()

    def insert_map_point(self, map_point):

        with self.data_mutex_:
            self.landmarks_[map_point.id_] = map_point
            self.active_landmarks_[map_point.id_] = map_point

    def get_all_map_points(self):
        with self.data_mutex_:
            return list(self.landmarks_.values())
        
    def get_all_keyframes(self):
        with self.data_mutex_:
            return list(self.keyframes_.values())
        
    def get_active_map_points(self):
        with self.data_mutex_:
            return list(self.active_landmarks_.values())
        
    def get_active_keyframes(self):
        with self.data_mutex_:
            return list(self.active_keyframes_.values())
        

    def clean_map(self):
        
        with self.data_mutex_:
            ids_to_remove = []

            for mp_id, map_point in self.landmarks_.items() : 
                if map_point.is_outlier_ or map_point.observed_times_ == 0:
                    ids_to_remove.append(mp_id)

            for mp_id in ids_to_remove:
                self.landmarks_.pop(mp_id, None)
                self.active_landmarks_.pop(mp_id, None)


    def remove_old_keyframe(self):

        if len(self.active_keyframes_) <= self.num_active_keyframes_:
            return
        min_id = min(self.active_keyframes_.keys())
        self.active_keyframes_.pop(min_id, None)
