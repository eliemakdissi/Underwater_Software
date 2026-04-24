import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2 as cv
import open3d as o3d

from cloud_sift import generate_cloud
from generate_dense_cloud import generate_dense_cloud

lowe = 0.7


CALIB_STEREO = 'SLAM/calibration/param/stereo_a_lenvers.pkl' 
with open(CALIB_STEREO, 'rb') as f:
    params = pickle.load(f)

K_rect = params['P1'][:3, :3]
dist_zero = np.zeros((4, 1))

print(K_rect)

PREVIOUS_PATH_IMG_L = f'SLAM/data_sortie_mer/frames/gauche/sortie_left/frames/frame_002409.jpg'
PREVIOUS_PATH_IMG_R = f'SLAM/data_sortie_mer/frames/droite/sortie_right/frames/frame_002409.jpg'
previous3d, previous2d, previous_desc = generate_cloud(PATH_IMG_L=PREVIOUS_PATH_IMG_L, PATH_IMG_R=PREVIOUS_PATH_IMG_R)
nuage_global = [previous3d]
pose_camera_globale = np.eye(4)

nuage_global_o3d = o3d.geometry.PointCloud()

matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

for i in range (2409,2420) : 
    
    CURRENT_PATH_IMG_L = f'SLAM/data_sortie_mer/frames/gauche/sortie_left/frames/frame_{i+1:06d}.jpg'
    CURRENT_PATH_IMG_R = f'SLAM/data_sortie_mer/frames/gauche/sortie_left/frames/frame_{i+1:06d}.jpg'
    print(CURRENT_PATH_IMG_L)
    current3d, current2d, current_desc = generate_cloud(PATH_IMG_L=CURRENT_PATH_IMG_L, PATH_IMG_R=CURRENT_PATH_IMG_R)

    # 1. Matching
    knn_matches = matcher.knnMatch(previous_desc, current_desc, k=2)

    # 2. Filtrage de Lowe 
    mask1 = []
    mask2 = []
    
    for match in knn_matches:
        if len(match) == 2:
            m, n = match
            if m.distance < n.distance * lowe:
                mask1.append(m.queryIdx)
                mask2.append(m.trainIdx)

    # 3. Extraction EXACTE des points qui ont matché
    pts_3D_pnp = np.array(previous3d[mask1], dtype=np.float32)
    pts_2D_pnp = np.array(current2d[mask2], dtype=np.float32)

    succes, rvec, tvec, inliers = cv.solvePnPRansac(objectPoints=pts_3D_pnp, imagePoints=pts_2D_pnp, cameraMatrix=K_rect, distCoeffs=dist_zero, iterationsCount=100,
                reprojectionError=3.0, flags=cv.SOLVEPNP_ITERATIVE)

    if succes : 
        distance_parcourue = np.linalg.norm(tvec)
        print(f"Frame {i+1} -> Déplacement estimé : {distance_parcourue:.4f} mètres")

        R, _ = cv.Rodrigues(rvec)
        T_local = np.eye(4)
        T_local[:3, :3] = R
        T_local[:3, 3] = tvec.flatten()

        pose_camera_globale = pose_camera_globale @ np.linalg.inv(T_local)

        points_denses, couleurs_denses = generate_dense_cloud(
            PATH_IMG_L=CURRENT_PATH_IMG_L, 
            PATH_IMG_R=CURRENT_PATH_IMG_R, 
            params=params, 
            pose_camera_globale=pose_camera_globale
        )

        pcd_frame = o3d.geometry.PointCloud()
        pcd_frame.points = o3d.utility.Vector3dVector(points_denses)
        pcd_frame.colors = o3d.utility.Vector3dVector(couleurs_denses)
        pcd_frame_down = pcd_frame.voxel_down_sample(voxel_size=0.02)
        nuage_global_o3d += pcd_frame_down

        points_homogenes = np.hstack((current3d, np.ones((len(current3d), 1))))
        current3d_aligne = (pose_camera_globale @ points_homogenes.T).T[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(current3d_aligne)
        pcd_down = pcd.voxel_down_sample(voxel_size=0.05)  # 5cm de résolution
        nuage_global.append(np.asarray(pcd_down.points))

        # On met à jour les "previous" uniquement si PnP a réussi !
        previous3d = current3d 
        previous2d = current2d
        previous_desc = current_desc

carte_finale_3D = np.vstack(nuage_global)



fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
print(len(carte_finale_3D))

# On filtre les points aberrants trop loin (ex: > 15 mètres)

mask = (carte_finale_3D[:, 2] > 0.05) & (carte_finale_3D[:, 2] < 5)
p3d = carte_finale_3D[mask]


ax.scatter(p3d[:, 0], p3d[:, 2], -p3d[:, 1], s=1, c='r') # On inverse Y et Z pour l'affichage
ax.set_xlabel('X (Largeur)')
ax.set_ylabel('Z (Profondeur)')
ax.set_zlabel('Y (Hauteur)')
plt.title("Aperçu rapide du nuage de points")
plt.show()








    

# Visualisation carte dense 
'''
transform_inverse_Y = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
nuage_global_o3d.transform(transform_inverse_Y)

o3d.visualization.draw_geometries([nuage_global_o3d], window_name="Carte 3D Dense du SLAM")

'''