#IDEE GlOBALE DE LA PIPELINE

#1 Preprocessing : preprocessing.py -> on essaye de travailler l'image et de l'améliorer
#2 Calibrage de la caméra avec photos + OpenCV
#3 Feature Detector : KAZE/AKAZE (partie théorique sur LateX)
#4 Feature matching : distance de Hamming avec BrutForce sur OpenCV + filtrage de Lowe
#5 RANSAC et Matrice Essentielle : on extrait le mouvement avec la matrice essentielle
#6 Extraction 3D : avec nos points on fait une triangulation

import pickle
import cv2 as cv
import numpy as np
import argparse
import matplotlib.pyplot as plt

import SLAM.brouillons.opérationnel.preprocess as preprocess
from calibration import calibration


# Args
# AKAZE
parser = argparse.ArgumentParser(description='AKAZE local features matching pour drone.')
parser.add_argument('--input1', help='Path to input image 1.', default='SLAM/real_camera_png/img1.jpg')
parser.add_argument('--input2', help='Path to input image 2.', default='SLAM/real_camera_png/img2.jpg')
parser.add_argument("--akaze", type=float, default=0.001, help="Seuil de détection AKAZE")
# Filtrage de Lowe
parser.add_argument('--lowe', help='Lowe ratio', type=float, default=0.8)
# RANSAC
parser.add_argument('--ransac', help='RANSAC threshold', type=float, default=1)

args = parser.parse_args()

img1 = cv.imread(args.input1)
img2 = cv.imread(args.input2)

# Preprocessing

img1 = preprocess.cl_correction(img1)
img2 = preprocess.cl_correction(img2)

# Calibration parameters - on utilisera ensuite la fonction de calibrage dans calibration.py

with open('SLAM/calibration/parametres_calib_actif.txt','rb') as f :
    mtx, dist, newcameramtx, roi = pickle.load(f)

img1 = calibration.cor_calib(img = img1, mtx=mtx, dist=dist, newcameramtx=newcameramtx, roi=roi)
img2 = calibration.cor_calib(img = img2, mtx=mtx, dist=dist, newcameramtx=newcameramtx, roi=roi)

if img1 is None or img2 is None:
    print('Erreur : Impossible de trouver ou d\'ouvrir les images ! Vérifie les chemins.')
    exit(0)

# Feature detection
akaze = cv.AKAZE_create(threshold= args.akaze)
kpts1, desc1 = akaze.detectAndCompute(img1, None) # None -> pas de mask
kpts2, desc2 = akaze.detectAndCompute(img2, None)

# Feature matching
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False) # Not robust enough for underwater environment
knn_matches = matcher.knnMatch(desc1, desc2, k=2)

good_matchs = []
pts1 = []
pts2 = []

for m,n in knn_matches: # m et n sont des instances de la classe DMatch
    if m.distance < n.distance*args.lowe:
        good_matchs.append(m)
        pts1.append(kpts1[m.queryIdx].pt)
        pts2.append(kpts2[m.trainIdx].pt)

pts1 = np.float32(pts1) # Calcul matrice essentielle
pts2 = np.float32(pts2)

inliers1 = []
inliers2 = []
final_good_matches = []

#  RANSAC avec la Matrice Essentielle

if len(good_matchs) > 10:

    E, mask = cv.findEssentialMat(pts1, pts2, newcameramtx, method=cv.RANSAC, prob=0.999, threshold=args.ransac)
    
    if E is not None:
        matchesMask = mask.ravel().tolist()
        for i, is_inlier in enumerate(matchesMask):
            if is_inlier == 1:
                inliers1.append(kpts1[good_matchs[i].queryIdx])
                inliers2.append(kpts2[good_matchs[i].trainIdx])
                final_good_matches.append(good_matchs[i])      
    else:
        print("Échec du calcul de la matrice essentielle.")

else:
    print("Pas assez de correspondances trouvées")

#res = cv.drawMatches(img1, kpts1, img2, kpts2, final_good_matches, None, matchColor=(0, 0, 0), singlePointColor=(255, 0, 0), flags=2)


print('--- Résultats AKAZE ---')
print('# Keypoints 1:                  \t', len(kpts1))
print('# Keypoints 2:                  \t', len(kpts2))
print('# Matches initiaux (Lowe):            \t', len(good_matchs))
print('# Inliers (Vrais points après RANSAC):\t', len(inliers1))
'''
cv.imshow('AKAZE Matches', res)
cv.waitKey(0) 
cv.destroyAllWindows()
'''


vis1_color = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
vis2_color = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

cv.drawKeypoints(vis1_color, inliers1, vis1_color, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.drawKeypoints(vis2_color, inliers2, vis2_color, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
combined_res = cv.hconcat([vis1_color, vis2_color])
combined_rgb = cv.cvtColor(combined_res, cv.COLOR_BGR2RGB)

plt.figure(figsize=(16, 8), dpi=100)
plt.imshow(combined_rgb)
plt.axis('off')
plt.tight_layout()
plt.show() 