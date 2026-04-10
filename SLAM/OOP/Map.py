# Implementation of the Map class
import threading

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
            self.active_keyframes_ = frame

        if len(self.active_keyframes_) > self.num_active_keyframes_:
            self.remove_old_keyframe()

    def insert_map_point(self, map_point):

        with self.data_mutex_:
            self.landmarks_[map_point.id_] = map_point
            self.active_landmarks_[map_point.id_] = map_point

    def get_all_map_points(self):

        with self.data_mutex_:
            return self.landmarks_.copy()
        
    def get_all_keyframes(self):

        with self.data_mutex_:
            return self.keyframes_.copy()
        
    def get_active_map_points(self):

        with self.data_mutex_:
            return self.active_landmarks_.copy()
        
    def get_active_keyframes(self):
        with self.data_mutex_:
            return self.active_keyframes_.copy()
        

    def clean_map(self):
        pass

    def remove_old_keyframe(self):
        pass