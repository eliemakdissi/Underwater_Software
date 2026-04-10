import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2 as cv
import open3d as o3d

from orb_test import generate_cloud

lowe = 0.4



CALIB_GAUCHE = '/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/calibration/param/parametres_calibeau_deuxiemecam.txt'
with open(CALIB_GAUCHE, 'rb') as f:
    _, _, newcameramtx1, _ = pickle.load(f)
dist1 = None
PREVIOUS_PATH_IMG_L = '/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/data_sortie_mer/frames/gauche/sortie_left.mp4_fixed/frames/frame_010932.jpg'
PREVIOUS_PATH_IMG_R = '/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/data_sortie_mer/frames/droite/sortie_right.mp4_fixed/frames/frame_010932.jpg'

previous3d, previous2d, previous_desc = generate_cloud(PATH_IMG_L=PREVIOUS_PATH_IMG_L, PATH_IMG_R=PREVIOUS_PATH_IMG_R)
nuage_global = [previous3d]

pose_camera_globale = np.eye(4)

matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

for i in range (10932,11000) : 
    
    CURRENT_PATH_IMG_L = f'/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/data_sortie_mer/frames/gauche/sortie_left.mp4_fixed/frames/frame_00{i+1:06d}.jpg'
    CURRENT_PATH_IMG_R = f'/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/data_sortie_mer/frames/droite/sortie_right.mp4_fixed/frames/frame_00{i+1:06d}.jpg'
    print("Traitement de :", CURRENT_PATH_IMG_L)

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

    # 4. Sécurité : Ne lancer PnP que si l'on a assez de points
    if len(pts_3D_pnp) < 4 or len(pts_2D_pnp) < 4:
        print("Pas assez de points pour lancer solvePnP")
        continue # On passe à la frame suivante au lieu de planter
        
    elif len(pts_3D_pnp) != len(pts_2D_pnp):
        print(" Erreur de correspondance.")
        continue

    # 5. Si tout est bon, on lance l'algo (ici on est en sécurité)
    succes, rvec, tvec, inliers = cv.solvePnPRansac(
        objectPoints=pts_3D_pnp, 
        imagePoints=pts_2D_pnp, 
        cameraMatrix=newcameramtx1, 
        distCoeffs=dist1, 
        flags=cv.SOLVEPNP_ITERATIVE
    )

    if succes and inliers is not None and len(inliers) > 10: 

        R, _ = cv.Rodrigues(rvec)
        T_local = np.eye(4)
        T_local[:3, :3] = R
        T_local[:3, 3] = tvec.flatten()

        pose_camera_globale = pose_camera_globale @ np.linalg.inv(T_local)

        points_homogenes = np.hstack((current3d, np.ones((len(current3d), 1))))
        current3d_aligne = (pose_camera_globale @ points_homogenes.T).T[:, :3]

        nuage_global.append(current3d_aligne)

        # On met à jour les "previous" uniquement si PnP a réussi !
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








    




