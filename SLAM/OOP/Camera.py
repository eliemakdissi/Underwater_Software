import cv2 as cv
import os
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


class StereoRecorder:
    """Records each (left, right) frame pair handed to write() into two
    separate MP4 files. Inter-pair sync is preserved by construction: pair
    N goes to frame index N in both files, so any desync visible at
    playback time is real and originates upstream of this recorder.

    Outputs land at <output_dir>/<prefix>_<timestamp>_left.mp4 and ..._right.mp4.
    Writers are opened lazily on the first write() so the recorder picks up
    the actual frame size automatically.
    """

    def __init__(self, output_dir="recordings", fps=30.0,
                 fourcc="mp4v", prefix="live"):
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path_l = os.path.join(output_dir, f"{prefix}_{ts}_left.mp4")
        self.path_r = os.path.join(output_dir, f"{prefix}_{ts}_right.mp4")
        self._fourcc = cv.VideoWriter_fourcc(*fourcc)
        self._fps = float(fps)
        self._writer_l = None
        self._writer_r = None
        self._frames_written = 0
        self._disabled = False

    def write(self, img_l, img_r):
        if self._disabled or img_l is None or img_r is None:
            return
        if self._writer_l is None:
            h_l, w_l = img_l.shape[:2]
            h_r, w_r = img_r.shape[:2]
            self._writer_l = cv.VideoWriter(self.path_l, self._fourcc,
                                            self._fps, (w_l, h_l))
            self._writer_r = cv.VideoWriter(self.path_r, self._fourcc,
                                            self._fps, (w_r, h_r))
            if not (self._writer_l.isOpened() and self._writer_r.isOpened()):
                print("[StereoRecorder] WARNING: failed to open VideoWriter "
                      "(codec/permissions); recording disabled.")
                self._writer_l = self._writer_r = None
                self._disabled = True
                return
            print(f"[StereoRecorder] Recording to {self.path_l} "
                  f"and {self.path_r} ({w_l}x{h_l} @ {self._fps:.1f} fps)")
        self._writer_l.write(img_l)
        self._writer_r.write(img_r)
        self._frames_written += 1

    def stop(self):
        if self._writer_l is not None:
            self._writer_l.release()
        if self._writer_r is not None:
            self._writer_r.release()
        if self._frames_written > 0:
            print(f"[StereoRecorder] Wrote {self._frames_written} frame "
                  f"pair(s) to {self.path_l} and {self.path_r}.")