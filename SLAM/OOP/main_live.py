import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

import cv2 as cv
import numpy as np
import time

from Map import Map
from Frame_without_rectify import Frame
from Frontend_without_rectify import Frontend
from ParamServer import start_param_server
from Camera import Camera, StereoRecorder #

# Toggle to save every (left, right) pair processed by SLAM into two MP4
# files under recordings/. Lets you check inter-camera desync afterwards
# by replaying the pair (e.g. with stereo_monitor.py).
RECORD_PAIRS = True
RECORD_DIR = "recordings"
RECORD_FPS = 30.0

BACKEND_TYPE = "g2o"

if BACKEND_TYPE == "gtsam":
    from Backend2 import Backend
else:
    from Backend import Backend

def save_point_cloud_ply(filename, xyz_points):
    print(f"Sauvegarde du nuage de points dans {filename}...")
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in xyz_points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print("Sauvegarde terminée !")

def main():
    print("Initialisation Live SLAM")

    #Init    
    
    slam_map = Map()
    slam_map.start_dash_server()
    slam_map.start_frame_server()
    start_param_server()

    backend = Backend(params_stereo=Frame.params, map=slam_map)
    frontend = Frontend(params_stereo_=Frame.params, map=slam_map, backend=backend)

    # pipe_l = "udpsrc port=5000 ! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink sync=false"
    # pipe_r = "udpsrc port=5001 ! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink sync=false"
    pipe_l = "SLAM/images_test/set_4_caillou/2026-04-15_14-52-14_left.mp4"
    pipe_r = "SLAM/images_test/set_4_caillou/2026-04-15_14-52-14_right.mp4"

    stream_l = Camera(pipe_l, "Cam_Gauche")
    stream_r = Camera(pipe_r, "Cam_Droite")
    time.sleep(2.0)

    recorder = (StereoRecorder(output_dir=RECORD_DIR, fps=RECORD_FPS)
                if RECORD_PAIRS else None)

    frame_id = 0
    print("\n--- Début du Tracking Live. Appuyez sur Ctrl+C dans le terminal pour arrêter et générer la carte. ---")

    try:
        # 3. Boucle temps réel
        while True:
            t_start = time.time()
            ret_l, img_l = stream_l.read()
            ret_r, img_r = stream_r.read()

            if not ret_l or not ret_r or img_l is None or img_r is None:
                time.sleep(0.01)
                continue

            if recorder is not None:
                recorder.write(img_l, img_r)

            # time.time() - timestamp absolu pour la Frame
            current_frame = Frame.create_frame(time_stamp=time.time(), left_img=img_l, right_img=img_r)
            success = frontend.add_frame(current_frame)

            if not success:
                print(f"Frame Live {frame_id:04d} | Tracking en échec ou initialisation en cours...")

            print(f"Frame Live {frame_id:04d} | Latence: {(time.time() - t_start)*1000:.1f}ms")
            frame_id += 1

    except KeyboardInterrupt:
        print("\nArrêt du flux Live demandé par l'utilisateur.")

    finally:
        print("\nFermeture des caméras...")
        stream_l.stop()
        stream_r.stop()
        if recorder is not None:
            recorder.stop()

        print("Génération de la carte 3D finale...")
        time.sleep(2.0)
        slam_map.build_mesh()
        points_3d = slam_map.get_all_map_points()

        if points_3d:
            xyz = np.array([mp.pos for mp in points_3d])
            mask = (xyz[:, 2] > 0.1) & (xyz[:, 2] < 15.0)
            xyz_clean = xyz[mask]
            save_point_cloud_ply("map_optimisee_live.ply", xyz_clean)
        else:
            print("Erreur : Aucun point valide dans la carte à exporter.")

        print(f"\n--- Processus terminé. Le dashboard Dash reste actif sur http://0.0.0.0:8050 ---")
        print("Faites Ctrl+C une deuxième fois pour tuer le serveur web.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Arrêt complet.")

if __name__ == '__main__':
    main()