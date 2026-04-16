import numpy as np
import cv2 as cv
import pickle
import time
import argparse
import os

def drawlines(img1, img2, lines, pts1, pts2):
    """
    Fonction utilitaire pour dessiner les lignes épipolaires et les points.
    img1: Image sur laquelle dessiner les lignes
    img2: Image contenant les points d'origine
    lines: Équations des lignes calculées par computeCorrespondEpilines
    pts1, pts2: Les points correspondants
    """
    r, c = img1.shape[:2]
    img1_color = img1.copy()
    img2_color = img2.copy()
    
    # Si les images sont en niveaux de gris, on les passe en couleur pour dessiner
    if len(img1_color.shape) == 2:
        img1_color = cv.cvtColor(img1_color, cv.COLOR_GRAY2BGR)
        img2_color = cv.cvtColor(img2_color, cv.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        
        # Dessine la ligne sur img1
        img1_color = cv.line(img1_color, (x0, y0), (x1, y1), color, 1)
        # Dessine le point d'origine sur img1
        img1_color = cv.circle(img1_color, tuple(np.int32(pt1)), 5, color, -1)
        # Dessine le point correspondant sur img2
        img2_color = cv.circle(img2_color, tuple(np.int32(pt2)), 5, color, -1)
        
    return img1_color, img2_color

def interactive_epipolar(PATH_IMG_L: str, PATH_IMG_R: str):
    print("Chargement des images et calcul de la Matrice Fondamentale...")
    
    img_l_brute = cv.imread(PATH_IMG_L) 
    img_r_brute = cv.imread(PATH_IMG_R)

    with open('SLAM/calibration/param/stereo_a_lenvers.pkl', 'rb') as f:
        params = pickle.load(f)

    # Undistort au niveau de l'image (ton choix précédent)
    img_l = cv.undistort(img_l_brute, params['mtx1'], params['dist1'], None, params['mtx1'])
    img_r = cv.undistort(img_r_brute, params['mtx2'], params['dist2'], None, params['mtx2'])

    # SIFT et calcul de F (Pour avoir une F robuste, on garde le processus classique)
    sift = cv.SIFT_create(nfeatures=1000)
    kp1, des1 = sift.detectAndCompute(img_l, None)
    kp2, des2 = sift.detectAndCompute(img_r, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    
    # Calcul de la matrice Fondamentale (La vraie star de l'épipolaire)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
    
    print(f"Matrice Fondamentale calculée avec {np.sum(mask)} inliers.")
    print("\n--- INSTRUCTIONS ---")
    print("1. Cliquez sur n'importe quelle image.")
    print("2. Une ligne épipolaire s'affichera sur l'autre image.")
    print("3. Appuyez sur 'c' pour effacer les lignes.")
    print("4. Appuyez sur 'q' pour quitter.")

    # --- PARTIE INTERACTIVE ---
    
    # Variables globales pour le clic
    global_pts1 = []
    global_pts2 = []
    
    # Copie pour l'affichage interactif
    disp_l = img_l.copy()
    disp_r = img_r.copy()

    def click_event(event, x, y, flags, param):
        nonlocal disp_l, disp_r, F
        
        if event == cv.EVENT_LBUTTONDOWN:
            # On détermine sur quelle image on a cliqué
            window_name = param
            pt = np.float32([[x, y]])
            
            if window_name == 'Image Gauche':
                # Clic à gauche : On trace la ligne à DROITE
                lines = cv.computeCorrespondEpilines(pt, 1, F)
                lines = lines.reshape(-1, 3)
                disp_r, disp_l = drawlines(disp_r, disp_l, lines, pt, pt)
            
            elif window_name == 'Image Droite':
                # Clic à droite : On trace la ligne à GAUCHE
                lines = cv.computeCorrespondEpilines(pt, 2, F)
                lines = lines.reshape(-1, 3)
                disp_l, disp_r = drawlines(disp_l, disp_r, lines, pt, pt)
                
            # Mise à jour des fenêtres
            cv.imshow('Image Gauche', disp_l)
            cv.imshow('Image Droite', disp_r)

    # Initialisation des fenêtres
    cv.namedWindow('Image Gauche', cv.WINDOW_NORMAL)
    cv.namedWindow('Image Droite', cv.WINDOW_NORMAL)
    
    cv.setMouseCallback('Image Gauche', click_event, 'Image Gauche')
    cv.setMouseCallback('Image Droite', click_event, 'Image Droite')

    cv.imshow('Image Gauche', disp_l)
    cv.imshow('Image Droite', disp_r)

    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Effacer tout
            disp_l = img_l.copy()
            disp_r = img_r.copy()
            cv.imshow('Image Gauche', disp_l)
            cv.imshow('Image Droite', disp_r)

    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gauche', default='SLAM/data_sortie_mer/frames/gauche/sortie_left/frames/frame_002410.jpg')
    parser.add_argument('--droite', default='SLAM/data_sortie_mer/frames/droite/sortie_right/frames/frame_002410.jpg')
    args = parser.parse_args()
    interactive_epipolar(args.gauche, args.droite)