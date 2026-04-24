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
        fps = self.cap.get(cv.CAP_PROP_FPS) if self.cap.isOpened() else 0.0
        period = 1.0 / fps if 1.0 < fps < 240.0 else 0.0
        if period > 0:
            print(f"[{self.name}] Source FPS={fps:.2f}, throttling decoder to real time.")
        next_tick = time.time()
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                with self._pose_mutex:
                    self.ret = ret
                    if ret:
                        self.frame = frame
                if period > 0:
                    next_tick += period
                    sleep_t = next_tick - time.time()
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                    else:
                        next_tick = time.time()

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