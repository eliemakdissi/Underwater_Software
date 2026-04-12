import threading
import numpy as np
import weakref

class MapPoint:
    _next_id = 0

    def __init__(self, id=None, position=None):

        self.id_ = id if id is not None else MapPoint._next_id
        self.pos_ = position if position is not None else np.zeros(3)
        self._data_mutex = threading.Lock()
        
        # État du point
        self.is_outlier_ = False
        self.observed_times_ = 0  # Nombre de fois que ce point a été vu
        
        # Feature qui voit ce point 3D (weakref)
        self.observations_ = []

    @property
    def pos(self):
        """ Getter thread-safe pour la position 3D """
        with self._data_mutex:
            return self.pos_

    @pos.setter
    def pos(self, new_pos):
        """ Setter thread-safe pour la position 3D """
        with self._data_mutex:
            self.pos_ = new_pos

    def add_observation(self, feature):
        """ Ajoute une référence vers une Feature qui voit ce point """
        with self._data_mutex:
            # On stocke une référence faible vers la feature
            self.observations_.append(weakref.ref(feature))
            self.observed_times_ += 1

    def get_obs(self):
        """ Récupère la liste des observations de manière sécurisée """
        with self._data_mutex:
            return self.observations_

    @staticmethod
    def create_new_mappoint(position):
        """ Factory method pour créer un nouveau point 3D """
        new_point = MapPoint(id=MapPoint._next_id, position=position)
        MapPoint._next_id += 1
        return new_point