import pickle
import cv2 as cv
import numpy as np
import argparse
import matplotlib.pyplot as plt

import preprocess as preprocess
# Plus besoin de 'calibration.py', on a le fichier .pkl complet !

# ==============================================================================
# 0. ARGUMENTS (On remplace frame t/t+1 par Gauche/Droite)
# ==============================================================================
parser = argparse.ArgumentParser(description='Pipeline Stéréo 3D avec AKAZE.')
parser.add_argument('--gauche', help='Chemin image gauche', default='SLAM/images_test/set_3_caillou/frame_0001_l.jpg')
parser.add_argument('--droite', help='Chemin image droite', default='SLAM/images_test/set_3_caillou/frame_0001_r.jpg')
parser.add_argument("--akaze", type=float, default=0.00001, help="Seuil de détection AKAZE")
parser.add_argument('--lowe', type=float, default=0.8, help='Ratio de Lowe')
parser.add_argument('--ytol', type=float, default=10.0, help='Tolérance horizontale (pixels)')
args = parser.parse_args()

img_g_brute = cv.imread(args.gauche)
img_d_brute = cv.imread(args.droite)


h, w = img_g_brute.shape[:2]
IMAGE_SIZE = (w, h)

print("1. Rectification des images...")
with open('SLAM/calibration/param/stereo_params_complets.pkl', 'rb') as f:
    params = pickle.load(f)

map1_x, map1_y = cv.initUndistortRectifyMap(params['mtx1'], params['dist1'], params['R1'], params['P1'], IMAGE_SIZE, cv.CV_32FC1)
map2_x, map2_y = cv.initUndistortRectifyMap(params['mtx2'], params['dist2'], params['R2'], params['P2'], IMAGE_SIZE, cv.CV_32FC1)

img_g_rect = cv.remap(img_g_brute, map1_x, map1_y, cv.INTER_LINEAR)
img_d_rect = cv.remap(img_d_brute, map2_x, map2_y, cv.INTER_LINEAR)

# ==============================================================================
# 2. PREPROCESSING (Améliorer les images sous-marines)
# ==============================================================================
print("2. Preprocessing...")
img_g_clean = preprocess.cl_correction(img_g_rect)
img_d_clean = preprocess.cl_correction(img_d_rect)

# ==============================================================================
# 3. FEATURE DETECTION (AKAZE)
# ==============================================================================
print("3. Extraction AKAZE...")
akaze = cv.AKAZE_create(threshold=args.akaze)
kpts_g, desc_g = akaze.detectAndCompute(img_g_clean, None)
kpts_d, desc_d = akaze.detectAndCompute(img_d_clean, None)

# ==============================================================================
# 4. MATCHING & FILTRAGE (Lowe + RANSAC classique)
# ==============================================================================
print("4. Feature Matching & Filtrage...")
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
knn_matches = matcher.knnMatch(desc_g, desc_d, k=2)

pts_g_brut = []
pts_d_brut = []
kpts_g_brut = []
kpts_d_brut = []

# --- 4.1 Filtre de Lowe ---
for m, n in knn_matches:
    if m.distance < n.distance * args.lowe:
        pts_g_brut.append(kpts_g[m.queryIdx].pt)
        pts_d_brut.append(kpts_d[m.trainIdx].pt)
        kpts_g_brut.append(kpts_g[m.queryIdx])
        kpts_d_brut.append(kpts_d[m.trainIdx])

pts_g_brut = np.float32(pts_g_brut)
pts_d_brut = np.float32(pts_d_brut)

print(f"Matchs initiaux après filtre de Lowe : {len(pts_g_brut)}")

# --- 4.2 Nettoyage géométrique avec RANSAC ---
inliers_g = []
inliers_d = []
pts_g = []
pts_d = []

if len(pts_g_brut) > 10:
    # On utilise cv.findFundamentalMat uniquement pour profiter de sa fonction RANSAC intégrée.
    # Le seuil de 3.0 est la tolérance de RANSAC (en pixels) par rapport au modèle global qu'il va trouver.
    F, mask = cv.findFundamentalMat(pts_g_brut, pts_d_brut, cv.FM_RANSAC, 8.0, 0.99)
    
    if mask is not None:
        mask = mask.ravel()
        for i in range(len(pts_g_brut)):
            if mask[i] == 1: # Si RANSAC valide ce point (Inlier)
                pts_g.append(pts_g_brut[i])
                pts_d.append(pts_d_brut[i])
                inliers_g.append(kpts_g_brut[i])
                inliers_d.append(kpts_d_brut[i])
                
        pts_g = np.float32(pts_g)
        pts_d = np.float32(pts_d)
    else:
        print("Erreur : RANSAC n'a pas réussi à trouver un modèle cohérent.")
else:
    print("Pas assez de correspondances après Lowe pour lancer RANSAC.")

print('--- Résultats AKAZE & Stéréo (Définitifs) ---')
print(f'# Keypoints Gauche:   \t {len(kpts_g)}')
print(f'# Keypoints Droite:   \t {len(kpts_d)}')
print(f'# Matchs valides (3D):\t {len(pts_g)}')

# ==============================================================================
# 5. EXTRACTION 3D (Triangulation)
# ==============================================================================
if len(pts_g) > 10:
    print("5. Triangulation 3D...")
    # On croise les rayons mathématiques
    points4D = cv.triangulatePoints(params['P1'], params['P2'], pts_g.T, pts_d.T)
    points3D = (points4D[:3, :] / points4D[3, :]).T # Conversion 4D -> 3D

    # EXPORT PLY DIRECT (Pour visualiser le nuage !)
    nom_fichier = 'nuage_instant_T.ply'
    with open(nom_fichier, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points3D)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, points3D, fmt='%f %f %f')
    print(f"🎉 Nuage 3D exporté sous : {nom_fichier}")

else:
    print("Pas assez de correspondances pour la 3D.")

# ==============================================================================
# 6. VISUALISATION INTERACTIVE (Vérification des points)
# ==============================================================================
vis_g = cv.cvtColor(img_g_clean, cv.COLOR_BGR2RGB)
vis_d = cv.cvtColor(img_d_clean, cv.COLOR_BGR2RGB)

cv.drawKeypoints(vis_g, inliers_g, vis_g, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.drawKeypoints(vis_d, inliers_d, vis_d, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

combined = cv.hconcat([vis_g, vis_d])

plt.figure(figsize=(16, 8))
plt.title(f"Points utilisés pour la 3D ({len(pts_g)} Inliers)")
plt.imshow(combined)
plt.axis('off')
plt.tight_layout()
plt.show()



from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# On limite l'affichage pour ne pas voir les points aberrants trop loin (ex: > 5 mètres)
mask = points3D[:, 2] < 5.0 
p3d = points3D[mask]

ax.scatter(p3d[:, 0], p3d[:, 2], -p3d[:, 1], s=1, c='r') # On inverse Y et Z pour l'affichage
ax.set_xlabel('X (Largeur)')
ax.set_ylabel('Z (Profondeur)')
ax.set_zlabel('Y (Hauteur)')
plt.title("Aperçu rapide du nuage de points")
plt.show()