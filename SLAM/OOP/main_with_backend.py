import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import os


from Frame import Frame
from Map import Map
from Frontend import Frontend
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
    backend = Backend(map=slam_map, params_stereo=Frame.params)
    frontend = Frontend(params_stereo_=Frame.params, map=slam_map, backend=backend)

    start_frame = 300
    end_frame = 400 
    
    for i in range(start_frame, end_frame):
        t_start = time.time()
   
        path_l = f'SLAM/images_test/image_caillou/frame_{i:04d}_l.jpg'
        path_r = f'SLAM/images_test/image_caillou/frame_{i:04d}_r.jpg'
        
        if not os.path.exists(path_l) or not os.path.exists(path_r):
             continue
        
        img_l = cv.imread(path_l, cv.IMREAD_COLOR) 
        img_r = cv.imread(path_r, cv.IMREAD_COLOR)

        current_frame = Frame.create_frame(time_stamp=float(i), left_img=img_l, right_img=img_r)
        success = frontend.add_frame(current_frame)

        if success:
            vis_img = img_l.copy()
            #On dessine les inliers
            inliers = [f for f in current_frame.features_left_ if f.map_point_ is not None]
            for feat in inliers:
                cv.circle(vis_img, (int(feat.position_.pt[0]), int(feat.position_.pt[1])), 2, (0, 255, 0), -1)
            
            cv.putText(vis_img, f"Frame: {i} | KFs: {len(slam_map.get_all_keyframes())}", 
                       (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.imshow("SLAM Live Tracking", vis_img)
            cv.waitKey(1)

            
        print(f"Frame {i:03d} | Latence: {(time.time() - t_start)*1000:.1f}ms")

    cv.destroyAllWindows()


    # Visualisation finale optimisée
    time.sleep(2.0) 

    points_3d = slam_map.get_all_map_points()
    keyframes = slam_map.get_all_keyframes()
    
    if not points_3d:
        print("Erreur : Aucun point dans la carte.")
        return

    # Extraction des points XYZ optimisés
    xyz = np.array([mp.pos for mp in points_3d])

    mask = (xyz[:, 2] > 0.1) & (xyz[:, 2] < 15.0)
    xyz_clean = xyz[mask]
    
    save_point_cloud_ply("map_optimisee.ply", xyz_clean)
    
    # Extraction de la trajectoire (Positions des Keyframes optimisées par g2o)
    kfs_sorted = sorted(keyframes, key=lambda x: x.id_)
    traj_pts = []
    for kf in kfs_sorted:
        # T_wc = inverse de la pose de la caméra
        T_wc = np.linalg.inv(kf.pose)
        traj_pts.append(T_wc[:3, 3])
    
    trajectory = np.array(traj_pts)

    # Plot 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Points de la carte (filtre de distance pour la propreté)
    mask = (xyz[:, 2] > 0.1) & (xyz[:, 2] < 15.0)
    ax.scatter(xyz[mask, 0], xyz[mask, 2], -xyz[mask, 1], s=1, c='red', alpha=0.3, label="Points 3D")
    
    # Trajectoire des Keyframes
    ax.plot(trajectory[:, 0], trajectory[:, 2], -trajectory[:, 1], c='blue', marker='o', label="Trajectoire Optimisée")

    ax.set_xlabel('X')
    ax.set_ylabel('Z (Profondeur)')
    ax.set_zlabel('Y (Hauteur)')
    ax.legend()
    plt.title(f"SLAM Final : {len(points_3d)} points, {len(keyframes)} Keyframes")
    plt.show()

if __name__ == '__main__':
    main()