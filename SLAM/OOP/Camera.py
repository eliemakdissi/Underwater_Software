import cv2 as cv
import threading
import time

class Camera:
    def __init__(self, pipeline_str, name="Camera"):
        self.name = name
        self.pipeline_str = pipeline_str
        
        self.cap = cv.VideoCapture(self.pipeline_str, cv.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print(f"Impossible d'ouvrir le flux {self.name}.")
            print(f"Pipeline utilisé : {self.pipeline_str}")
            
        self.ret = False
        self.frame = None
        self._pose_mutex = threading.Lock() 
        self.running = True
        
        # Démarrage automatique du thread de capture
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        print(f"[{self.name}] Flux démarré.")

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                with self._pose_mutex:
                    self.ret = ret
                    if ret:
                        self.frame = frame

    def read(self): # Fonction qu'appelle le SLAM
        with self._pose_mutex:
            if self.ret and self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()
        print(f"[{self.name}] Flux arrêté.")