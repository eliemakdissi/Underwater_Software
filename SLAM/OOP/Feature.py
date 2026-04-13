# Implementation of the Feature class

import weakref

class Feature:
    def __init__(self, frame=None, keypoint=None, descriptor = None):
        """
        Représente un point d'intérêt extrait d'une image.
        """
        # eviter le blocage d'une supression
        self.frame_ = weakref.ref(frame) if frame is not None else None
        
        # objet avec .pt, .size dans cv2
        self.position_ = keypoint
        self.descriptor_ = descriptor
        
        # point 3d correspondant au feature, weakref aussi si on veut supprimer le point
        self.map_point_ = None 
        self.is_outlier_ = False          
        self.is_on_left_image_ = True     

