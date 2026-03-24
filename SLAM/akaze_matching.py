from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='AKAZE local features matching pour drone.')
parser.add_argument('--input1', help='Path to input image 1.', default='image1.jpeg')
parser.add_argument('--input2', help='Path to input image 2.', default='image2.jpeg')
parser.add_argument(
        "--akaze-threshold", 
        type=float, 
        default=0.001, 
        help="Seuil de détection AKAZE. Diminuer (ex: 0.0005) pour trouver PLUS de points."
    )
args = parser.parse_args()

# 2. On charge les images DIRECTEMENT (sans samples.findFile)
img1 = cv.imread(args.input1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(args.input2, cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print('Erreur : Impossible de trouver ou d\'ouvrir les images ! Vérifie les chemins.')
    exit(0)

# 3. Détection AKAZE
akaze = cv.AKAZE_create(threshold=args.akaze_threshold)
kpts1, desc1 = akaze.detectAndCompute(img1, None)
kpts2, desc2 = akaze.detectAndCompute(img2, None)

# 4. Matching brut (K-Nearest Neighbors)
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
nn_matches = matcher.knnMatch(desc1, desc2, 2)

# 5. Filtrage par le test de ratio de Lowe
good_matches = []
pts1 = []
pts2 = []
nn_match_ratio = 0.8 

for m, n in nn_matches:
    if m.distance < nn_match_ratio * n.distance:
        good_matches.append(m)
        pts1.append(kpts1[m.queryIdx].pt)
        pts2.append(kpts2[m.trainIdx].pt)

# On convertit les listes en tableaux Numpy pour les calculs mathématiques
pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

inliers1 = []
inliers2 = []
final_good_matches = []

# 6. Ransac
if len(good_matches) > 10:
    # On calcule la matrice d'homographie et on récupère le masque des "bons" points
    matrix, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    
    for i, is_inlier in enumerate(matchesMask):
        if is_inlier == 1:
            inliers1.append(kpts1[good_matches[i].queryIdx])
            inliers2.append(kpts2[good_matches[i].trainIdx])
            final_good_matches.append(good_matches[i])
else:
    print("Pas assez de correspondances trouvées !")

res = cv.drawMatches(img1, kpts1, img2, kpts2, final_good_matches, None, 
                     matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=2)

cv.imwrite("akaze_result.png", res)

print('--- Résultats du Matching AKAZE + RANSAC ---')
print('# Keypoints Image 1:                  \t', len(kpts1))
print('# Keypoints Image 2:                  \t', len(kpts2))
print('# Matches initiaux (Lowe):            \t', len(good_matches))
print('# Inliers (Vrais points après RANSAC):\t', len(inliers1))


cv.imshow('AKAZE Matches', res)
cv.waitKey(0) 
cv.destroyAllWindows()