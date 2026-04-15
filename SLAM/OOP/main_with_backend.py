import cv2 as cv
import numpy as np
import time
import os

from Frame import Frame
from Map import Map
from Frontend import Frontend

# ---- Choose backend: "g2o" or "gtsam" ----
BACKEND_TYPE = "g2o"

if BACKEND_TYPE == "gtsam":
    from Backend2 import Backend
else:
    from Backend import Backend

def save_point_cloud_ply(filename, xyz_points):
    """
    Sauvegarde un tableau numpy de points 3D au format .ply ASCII.
    """
    print(f"💾 Sauvegarde du nuage de points dans {filename}...")
    with open(filename, 'w') as f:
        # En-tête strict du format PLY
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        # Écriture des coordonnées
        for p in xyz_points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
    print("Sauvegarde terminée !")
def main():
    print("initialisation slam")
    

    slam_map = Map()
    slam_map.start_dash_server()

    if BACKEND_TYPE == "gtsam":
        backend = Backend(params_stereo=Frame.params, slam_map=slam_map)
    else:
        backend = Backend(map=slam_map, params_stereo=Frame.params)

    frontend = Frontend(params_stereo_=Frame.params, map=slam_map, backend=backend)

    start_frame = 20
    end_frame = 250

    for i in range(start_frame, end_frame):
        t_start = time.time()

        path_l = f'SLAM/images_test/set_4_caillou/frame_{i:04d}_l.jpg'
        path_r = f'SLAM/images_test/set_4_caillou/frame_{i:04d}_r.jpg'
        
        if not os.path.exists(path_l) or not os.path.exists(path_r):
             continue
        
        img_l = cv.imread(path_l, cv.IMREAD_COLOR) 
        img_r = cv.imread(path_r, cv.IMREAD_COLOR)

        current_frame = Frame.create_frame(time_stamp=float(i), left_img=img_l, right_img=img_r)
        success = frontend.add_frame(current_frame)

        if not success:
            print('not success')

        print(f"Frame {i:03d} | Latence: {(time.time() - t_start)*1000:.1f}ms")


    # Visualisation finale optimisée
    time.sleep(2.0)

    points_3d = slam_map.get_all_map_points()

    if not points_3d:
        print("Erreur : Aucun point dans la carte.")
        return

    # Extraction des points XYZ optimisés
    xyz = np.array([mp.pos for mp in points_3d])

    mask = (xyz[:, 2] > 0.1) & (xyz[:, 2] < 15.0)
    xyz_clean = xyz[mask]

    save_point_cloud_ply("map_optimisee.ply", xyz_clean)

    print(f"\n--- Processing complete. Live plot at http://0.0.0.0:8050 ---")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()