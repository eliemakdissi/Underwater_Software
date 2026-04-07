import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2 as cv
import open3d as o3d

from generate_cloud import generate_cloud

lowe = 0.8



CALIB_GAUCHE = 'SLAM/calibration/param/parametres_calibeau_deuxiemecam.txt'
with open(CALIB_GAUCHE, 'rb') as f:
    mtx1, dist1, _, _ = pickle.load(f)

PREVIOUS_PATH_IMG_L = f'SLAM/images_test/set_3_caillou/frame_0002_l.jpg'
PREVIOUS_PATH_IMG_R = f'SLAM/images_test/set_3_caillou/frame_0002_r.jpg'
previous3d, previous2d, previous_desc = generate_cloud(PATH_IMG_L=PREVIOUS_PATH_IMG_L, PATH_IMG_R=PREVIOUS_PATH_IMG_R)
nuage_global = [previous3d]

pose_camera_globale = np.eye(4)

matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

for i in range (2,14) : 
    
    CURRENT_PATH_IMG_L = f'SLAM/images_test/set_3_caillou/frame_{i+1:04d}_l.jpg'
    CURRENT_PATH_IMG_R = f'SLAM/images_test/set_3_caillou/frame_{i+1:04d}_r.jpg'
    current3d, current2d, current_desc = generate_cloud(PATH_IMG_L=CURRENT_PATH_IMG_L, PATH_IMG_R=CURRENT_PATH_IMG_R)

    # Matching
    knn_matches = matcher.knnMatch(previous_desc, current_desc, k=2)

    # Filtrage de Lowe 
    mask1 = []
    mask2 = []
    
    for match in knn_matches:
        if len(match)==2:
            m,n = match
            if m.distance < n.distance * lowe:
                mask1.append(m.queryIdx)
                mask2.append(m.trainIdx)

    
    pts_3D_pnp = previous3d[mask1]
    pts_2D_pnp = current2d[mask2]

    succes, rvec, tvec, inliers = cv.solvePnPRansac(objectPoints=pts_3D_pnp, imagePoints=pts_2D_pnp, cameraMatrix=mtx1, distCoeffs=dist1, flags=cv.SOLVEPNP_ITERATIVE)

    if succes : 

        R, _ = cv.Rodrigues(rvec)
        T_local = np.eye(4)
        T_local[:3, :3] = R
        T_local[:3, 3] = tvec.flatten()

        pose_camera_globale = pose_camera_globale @ np.linalg.inv(T_local)

        points_homogenes = np.hstack((current3d, np.ones((len(current3d), 1))))
        current3d_aligne = (pose_camera_globale @ points_homogenes.T).T[:, :3]

        nuage_global.append(current3d_aligne)

        previous3d = current3d 
        previous2d = current2d
        previous_desc = current_desc

carte_finale_3D = np.vstack(nuage_global)



fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


# On filtre les points aberrants trop loin (ex: > 15 mètres)

mask = (carte_finale_3D[:, 2] > 0.05) & (carte_finale_3D[:, 2] < 5)
p3d = carte_finale_3D[mask]


ax.scatter(p3d[:, 0], p3d[:, 2], -p3d[:, 1], s=1, c='r') # On inverse Y et Z pour l'affichage
ax.set_xlabel('X (Largeur)')
ax.set_ylabel('Z (Profondeur)')
ax.set_zlabel('Y (Hauteur)')
plt.title("Aperçu rapide du nuage de points")
plt.show()








    




