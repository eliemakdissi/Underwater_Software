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

    print(params['dist1'], params['dist2'])
    # UNDISTORT UNIQUEMENT
    img_l_undist = cv.undistort(img_l_brute, params['mtx1'], params['dist1'], params['mtx1'])
    img_r_undist = cv.undistort(img_r_brute, params['mtx2'], params['dist2'], params['mtx2'])

    img_l_clean = preprocess.cl_vert(img_l_undist)
    img_r_clean = preprocess.cl_vert(img_r_undist)

    
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    img_l_clean = clahe.apply(img_l_clean)
    img_r_clean = clahe.apply(img_r_clean)



    # FEATURE DETECTION SIFT
    #sift = cv.SIFT_create()
    sift = cv.SIFT_create(contrastThreshold=0.01)
    kpts_l, desc_l = sift.detectAndCompute(img_l_clean, None)
    kpts_r, desc_r = sift.detectAndCompute(img_r_clean, None)

    start_time = time.time()

    # MATCHING GLOBAL
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

    kpts_g_dessin = [cv.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts_g_brut]
    kpts_d_dessin = [cv.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts_d_brut]

    # On crée des DMatch qui disent "Relie l'index i à l'index i"
    matches_pour_dessin = [cv.DMatch(i, i, 0) for i in range(len(pts_g_brut))]

    # On dessine !
    img_matches = cv.drawMatches(
        img_l_clean, kpts_g_dessin, 
        img_r_clean, kpts_d_dessin, 
        matches_pour_dessin, None, 
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, 
        matchColor=(0, 255, 0)
    )

    # On affiche dans une fenêtre redimensionnée
    img_matches_resized = cv.resize(img_matches, (0, 0), fx=0.5, fy=0.5)
    cv.imshow("Apercu des Matchs (Lignes)", img_matches_resized)
    cv.waitKey(0) # Le script s'arrête ici jusqu'à ce que tu appuies sur une touche
    cv.destroyAllWindows()
    # FILTRAGE RANSAC
    pts_g = []
    pts_d = []

    if len(pts_g_brut) > 10:
        F, mask = cv.findFundamentalMat(pts_g_brut, pts_d_brut, cv.FM_RANSAC, 3.0, 0.99)
        if mask is not None:
            mask = mask.ravel() == 1
            pts_g = pts_g_brut[mask]
            pts_d = pts_d_brut[mask]


    kpts_g_dessin = [cv.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts_g]
    kpts_d_dessin = [cv.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts_d]

    # On crée des DMatch qui disent "Relie l'index i à l'index i"
    matches_pour_dessin = [cv.DMatch(i, i, 0) for i in range(len(pts_g))]

    # On dessine !
    img_matches = cv.drawMatches(
        img_l_clean, kpts_g_dessin, 
        img_r_clean, kpts_d_dessin, 
        matches_pour_dessin, None, 
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, 
        matchColor=(0, 255, 0)
    )

    # 4. On affiche dans une fenêtre redimensionnée
    img_matches_resized = cv.resize(img_matches, (0, 0), fx=0.5, fy=0.5)
    cv.imshow("Apercu des Matchs (Lignes)", img_matches_resized)
    cv.waitKey(0) # Le script s'arrête ici jusqu'à ce que tu appuies sur une touche
    cv.destroyAllWindows()

    end_time = time.time()
    
    print('--- Résultats SIFT ---')
    print(f'Paramètres sift :     \t {0.01:.4f}')
    print(f'Temps de calcul :     \t {end_time - start_time:.4f} sec')
    print(f'# Matchs (Lowe):      \t {len(pts_g_brut)}')
    print(f'# Matchs (RANSAC):    \t {len(pts_g)}')

    # ==========================================
    # 5. DEBUG VISUEL INTERACTIF (ZOOM SYNCHRONISÉ)
    # ==========================================
    if len(pts_g) > 0:
        # Création de 2 sous-graphiques avec axes liés (sharex, sharey)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        
        # Conversion BGR -> RGB pour Matplotlib
        img_l_rgb = cv.cvtColor(img_l_clean, cv.COLOR_BGR2RGB)
        img_r_rgb = cv.cvtColor(img_r_clean, cv.COLOR_BGR2RGB)

        # Tracé Image Gauche
        ax1.imshow(img_l_rgb)
        ax1.scatter(pts_g[:, 0], pts_g[:, 1], facecolors='none', edgecolors='lime', s=30, linewidths=1.5)
        ax1.set_title("Image Gauche")
        ax1.axis('off')

        # Tracé Image Droite
        ax2.imshow(img_r_rgb)
        ax2.scatter(pts_d[:, 0], pts_d[:, 1], facecolors='none', edgecolors='lime', s=30, linewidths=1.5)
        ax2.set_title("Image Droite")
        ax2.axis('off')

        plt.suptitle(f"Points Inliers ({len(pts_g)})\nZoomez sur l'une des images avec la loupe, l'autre suivra automatiquement !", fontsize=14)
        plt.tight_layout()
        plt.show()
    # ==========================================

    # 6. TRIANGULATION ET 3D
    p3d = []
    
    if len(pts_g) > 10:
        # P_left = K1 * [I | 0]
        P1_unrect = params['mtx1'] @ np.hstack((np.eye(3), np.zeros((3, 1))))
        
        # P_right = K2 * [R | T]
        P2_unrect = params['mtx2'] @ np.hstack((params['R'], params['T']))

        points4D = cv.triangulatePoints(P1_unrect, P2_unrect, pts_g.T, pts_d.T)
        points3D = (points4D[:3, :] / points4D[3, :]).T

        mask_z = (points3D[:, 2] > 0.05) & (points3D[:, 2] < 15.0)
        p3d = points3D[mask_z]
        
        if len(p3d) > 0:
            save_point_cloud_ply(os.path.join(DOSSIER_SORTIE, 'test_sift_unrect.ply'), p3d)
            
            fig_3d = plt.figure(figsize=(10, 8))
            ax = fig_3d.add_subplot(111, projection='3d')
            ax.scatter(p3d[:, 0], p3d[:, 2], -p3d[:, 1], s=2, c='red')
            ax.set_xlabel('X (Largeur)')
            ax.set_ylabel('Z (Profondeur)')
            ax.set_zlabel('Y (Hauteur)')
            plt.title(f"Nuage de points 3D ({len(p3d)} points valides)")
            plt.show()

    return p3d

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Pipeline Stéréo 3D avec SIFT.')
    parser.add_argument('--gauche', help='Chemin image gauche', default='SLAM/images_test/image_caillou/frame_0001_l.jpg')
    parser.add_argument('--droite', help='Chemin image droite', default='SLAM/images_test/image_caillou/frame_0001_r.jpg')
    args = parser.parse_args()
    generate_cloud(args.gauche, args.droite)