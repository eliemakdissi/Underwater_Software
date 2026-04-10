import numpy as np
import cv2 as cv
import glob
import pickle


CHESSBOARD_SIZE = (9, 7) 
SQUARE_SIZE = 0.030

PATH_IMAGES_GAUCHE = 'SLAM/calibration/set_doublecam/*_l.jpg'
PATH_IMAGES_DROITE = 'SLAM/calibration/set_doublecam/*_r.jpg'

CALIB_DROITE = 'SLAM/calibration/param/parametres_calibeau_premierecam.txt'
CALIB_GAUCHE = 'SLAM/calibration/param/parametres_calibeau_deuxiemecam.txt'

with open(CALIB_GAUCHE, 'rb') as f:
    mtx1, dist1, _, _ = pickle.load(f) # On ne garde que la matrice et la distorsion

with open(CALIB_DROITE, 'rb') as f:
    mtx2, dist2, _, _ = pickle.load(f)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Grille théorique
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE 

objpoints = [] # Points 3D dans le monde réel
imgpoints_gauche = [] # Points 2D sur l'image gauche
imgpoints_droite = [] # Points 2D sur l'image droite


images_gauche = sorted(glob.glob(PATH_IMAGES_GAUCHE))
images_droite = sorted(glob.glob(PATH_IMAGES_DROITE))

if len(images_gauche) != len(images_droite) or len(images_gauche) == 0:
    exit()

print(f"Analyse de {len(images_gauche)} paires d'images en cours...")
paires_valides = 0

for img_g_path, img_d_path in zip(images_gauche, images_droite):
    img_g = cv.imread(img_g_path)
    img_d = cv.imread(img_d_path)
    
    gray_g = cv.cvtColor(img_g, cv.COLOR_BGR2GRAY)
    gray_d = cv.cvtColor(img_d, cv.COLOR_BGR2GRAY)

    # On cherche le damier sur les DEUX images
    ret_g, corners_g = cv.findChessboardCorners(gray_g, CHESSBOARD_SIZE, None)
    ret_d, corners_d = cv.findChessboardCorners(gray_d, CHESSBOARD_SIZE, None)


    if ret_g == True and ret_d == True:

        corners2_g = cv.cornerSubPix(gray_g, corners_g, (11, 11), (-1, -1), criteria)
        corners2_d = cv.cornerSubPix(gray_d, corners_d, (11, 11), (-1, -1), criteria)
        
        #Empeche les inversements
        # Dist point 1 gauche et point 1 droit
        dist_normale = np.linalg.norm(corners2_g[0] - corners2_d[0])
        # Dist point 1 gauche et dernier point droit
        dist_inverse = np.linalg.norm(corners2_g[0] - corners2_d[-1])
        
        if dist_inverse < dist_normale:
            corners2_d = corners2_d[::-1]


        paires_valides += 1
        objpoints.append(objp)
        imgpoints_gauche.append(corners2_g)
        imgpoints_droite.append(corners2_d)

print(f"{paires_valides} paires valides trouvées sur {len(images_gauche)}.")

if paires_valides < 10:
    print("Less than 10 found")


image_size = gray_g.shape[::-1] 

flags = cv.CALIB_FIX_INTRINSIC

ret_stereo, mtx1, dist1, mtx2, dist2, R, T, E, F = cv.stereoCalibrate(
    objpoints, imgpoints_gauche, imgpoints_droite, 
    mtx1, dist1, mtx2, dist2, 
    image_size, criteria=criteria, flags=flags)

print(f"Erreur de reprojection RMS : {ret_stereo:.4f} pixels")
print(f"Vecteur de Translation T (Baseline en mètres) :\n{T}")


R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
    mtx1, dist1, mtx2, dist2, 
    image_size, R, T, 
    flags=cv.CALIB_ZERO_DISPARITY, alpha=0)


stereo_params = {
    'mtx1': mtx1, 'dist1': dist1,
    'mtx2': mtx2, 'dist2': dist2,
    'R': R, 'T': T,
    'R1': R1, 'R2': R2, 
    'P1': P1, 'P2': P2, 
    'roi1': roi1, 'roi2': roi2,
    'Q': Q
}

fichier_sortie = 'SLAM/calibration/param/stereo_params_complets.pkl'
with open(fichier_sortie, 'wb') as f:
    pickle.dump(stereo_params, f)
