import preprocess 
import numpy as np
import cv2 as cv
import pickle
import time
import argparse
import matplotlib.pyplot as plt
import os

DOSSIER_SORTIE = "SLAM/test_sift_ply"
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

def save_point_cloud_ply(filename, xyz_points):
    print(f"Sauvegarde du nuage de points dans {filename}...")
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz_points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in xyz_points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print("Sauvegarde terminée !")

def generate_cloud(PATH_IMG_L: str, PATH_IMG_R: str, lowe: float = 0.85):

    img_l_brute = cv.imread(PATH_IMG_L) 
    img_r_brute = cv.imread(PATH_IMG_R)

    with open('SLAM/calibration/param/stereo_a_lenvers.pkl', 'rb') as f:
        params = pickle.load(f)

    print("Distorsion Gauche:", params['dist1'].flatten())
    print("Distorsion Droite:", params['dist2'].flatten())

    # 1. PAS DE UNDISTORT SUR L'IMAGE (On garde les pixels originaux exacts)
    img_l_clean = preprocess.cl_vert(img_l_brute)
    img_r_clean = preprocess.cl_vert(img_r_brute)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_l_clean = clahe.apply(img_l_clean)
    img_r_clean = clahe.apply(img_r_clean)

    # 2. FEATURE DETECTION SIFT
    sift = cv.SIFT_create(contrastThreshold=0.01)
    kpts_l, desc_l = sift.detectAndCompute(img_l_clean, None)
    kpts_r, desc_r = sift.detectAndCompute(img_r_clean, None)

    start_time = time.time()

    # 3. MATCHING GLOBAL
    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(desc_l, desc_r, k=2)

    pts_g_brut = []
    pts_d_brut = []

    for m, n in knn_matches:
        if m.distance < n.distance * lowe:
            pts_g_brut.append(kpts_l[m.queryIdx].pt)
            pts_d_brut.append(kpts_r[m.trainIdx].pt)

    pts_g_brut = np.float32(pts_g_brut)
    pts_d_brut = np.float32(pts_d_brut)

    # --- DESSIN DES MATCHS BRUTS ---
    kpts_g_dessin = [cv.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts_g_brut]
    kpts_d_dessin = [cv.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts_d_brut]
    matches_pour_dessin = [cv.DMatch(i, i, 0) for i in range(len(pts_g_brut))]

    img_matches = cv.drawMatches(
        img_l_clean, kpts_g_dessin, 
        img_r_clean, kpts_d_dessin, 
        matches_pour_dessin, None, 
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0)
    )
    img_matches_resized = cv.resize(img_matches, (0, 0), fx=0.5, fy=0.5)
    cv.imshow("Apercu des Matchs AVANT Filtrage (Lignes)", img_matches_resized)
    cv.waitKey(0) 
    cv.destroyAllWindows()


    # 4. UNDISTORT DES COORDONNÉES ET FILTRAGE RANSAC (MATRICE ESSENTIELLE)
    pts_g_plot = []
    pts_d_plot = []
    pts_g_norm_good = []
    pts_d_norm_good = []

    if len(pts_g_brut) > 10:
        # Passage des pixels en coordonnées normalisées (idéales)
        pts_g_norm = cv.undistortPoints(pts_g_brut, params['mtx1'], params['dist1']).reshape(-1, 2)
        pts_d_norm = cv.undistortPoints(pts_d_brut, params['mtx2'], params['dist2']).reshape(-1, 2)

        # La tolérance RANSAC (ex: 4 pixels) doit être convertie à l'échelle normalisée
        seuil_pixels = 4.0 
        seuil_norm = seuil_pixels / params['mtx1'][0, 0]

        # Calcul de la Matrice Essentielle
        E, mask = cv.findEssentialMat(
            pts_g_norm, pts_d_norm, 
            focal=1.0, pp=(0., 0.), 
            method=cv.RANSAC, prob=0.99, threshold=seuil_norm
        )
        
        if mask is not None:
            mask = mask.ravel() == 1
            # On garde les points normalisés pour la triangulation mathématique
            pts_g_norm_good = pts_g_norm[mask]
            pts_d_norm_good = pts_d_norm[mask]
            
            # On garde les pixels d'origine pour l'affichage visuel
            pts_g_plot = pts_g_brut[mask]
            pts_d_plot = pts_d_brut[mask]

    # --- DESSIN DES MATCHS INLIERS ---
    kpts_g_dessin_inliers = [cv.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts_g_plot]
    kpts_d_dessin_inliers = [cv.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts_d_plot]
    matches_pour_dessin_inliers = [cv.DMatch(i, i, 0) for i in range(len(pts_g_plot))]

    img_matches_inliers = cv.drawMatches(
        img_l_clean, kpts_g_dessin_inliers, 
        img_r_clean, kpts_d_dessin_inliers, 
        matches_pour_dessin_inliers, None, 
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0)
    )
    img_matches_inliers_resized = cv.resize(img_matches_inliers, (0, 0), fx=0.5, fy=0.5)
    cv.imshow("Apercu des Matchs APRES Filtrage E (Lignes)", img_matches_inliers_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()

    end_time = time.time()
    
    print('--- Résultats SIFT ---')
    print(f'Temps de calcul :     \t {end_time - start_time:.4f} sec')
    print(f'# Matchs bruts (Lowe): \t {len(pts_g_brut)}')
    print(f'# Matchs (RANSAC E):  \t {len(pts_g_plot)}')


    # ==========================================
    # 5. DEBUG VISUEL INTERACTIF (ZOOM SYNCHRONISÉ)
    # ==========================================
    if len(pts_g_plot) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        img_l_rgb = cv.cvtColor(img_l_clean, cv.COLOR_GRAY2RGB)
        img_r_rgb = cv.cvtColor(img_r_clean, cv.COLOR_GRAY2RGB)

        ax1.imshow(img_l_rgb)
        ax1.scatter(pts_g_plot[:, 0], pts_g_plot[:, 1], facecolors='none', edgecolors='lime', s=30, linewidths=1.5)
        ax1.set_title("Image Gauche")
        ax1.axis('off')

        ax2.imshow(img_r_rgb)
        ax2.scatter(pts_d_plot[:, 0], pts_d_plot[:, 1], facecolors='none', edgecolors='lime', s=30, linewidths=1.5)
        ax2.set_title("Image Droite")
        ax2.axis('off')

        plt.suptitle(f"Points Inliers ({len(pts_g_plot)})\nZoomez sur l'une des images avec la loupe, l'autre suivra automatiquement !", fontsize=14)
        plt.tight_layout()
        plt.show()

    # ==========================================
    # 6. TRIANGULATION PURE ET 3D
    # ==========================================
    p3d = []
    
    if len(pts_g_norm_good) > 10:
        # Matrices de projections idéales (sans la distorsion et sans K)
        P1_ideal = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2_ideal = np.hstack((params['R'], params['T']))

        # On triangule les points normalisés !
        points4D = cv.triangulatePoints(P1_ideal, P2_ideal, pts_g_norm_good.T, pts_d_norm_good.T)
        points3D = (points4D[:3, :] / points4D[3, :]).T

        # Filtre de sécurité basique
        mask_z = (points3D[:, 2] > 0.05) & (points3D[:, 2] < 15.0)
        p3d = points3D[mask_z]
        
        if len(p3d) > 0:
            save_point_cloud_ply(os.path.join(DOSSIER_SORTIE, 'test_sift_ideal.ply'), p3d)
            
            fig_3d = plt.figure(figsize=(10, 8))
            ax = fig_3d.add_subplot(111, projection='3d')
            ax.scatter(p3d[:, 0], p3d[:, 2], -p3d[:, 1], s=2, c='red')
            ax.set_xlabel('X (Largeur)')
            ax.set_ylabel('Z (Profondeur)')
            ax.set_zlabel('Y (Hauteur)')
            plt.title(f"Nuage de points 3D mathématique ({len(p3d)} points valides)")
            plt.show()

    return p3d

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Pipeline Stéréo 3D avec SIFT.')
    parser.add_argument('--gauche', help='Chemin image gauche', default='SLAM/images_test/image_caillou/frame_0001_l.jpg')
    parser.add_argument('--droite', help='Chemin image droite', default='SLAM/images_test/image_caillou/frame_0001_r.jpg')
    args = parser.parse_args()
    generate_cloud(args.gauche, args.droite)