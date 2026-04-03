# Generate a 3D cloud of points

import preprocess 
import numpy as np
import cv2 as cv
import pickle
import time
import argparse
import matplotlib.pyplot as plt




def generate_cloud(PATH_IMG_L : str, PATH_IMG_R : str, akaze_t : float = 0.0001, lowe : int = 0.8) :

    img_l_brute = cv.imread(PATH_IMG_L)
    
    img_r_brute = cv.imread(PATH_IMG_R)
    
    h, w = img_l_brute.shape[:2]
    IMAGE_SIZE = (w, h)

    print(IMAGE_SIZE)

    with open('/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/calibration/param/stereo_params_complets.pkl', 'rb') as f:
        params = pickle.load(f)

    # Preprocessing
    
    mapl_x, mapl_y = cv.initUndistortRectifyMap(params['mtx1'], params['dist1'], params['R1'], params['P1'], IMAGE_SIZE, cv.CV_32FC1)
    mapr_x, mapr_y = cv.initUndistortRectifyMap(params['mtx2'], params['dist2'], params['R2'], params['P2'], IMAGE_SIZE, cv.CV_32FC1)

    print('map done')

    img_l_rect = cv.remap(img_l_brute, mapl_x, mapl_y, cv.INTER_LINEAR)
    img_r_rect = cv.remap(img_r_brute, mapr_x, mapr_y, cv.INTER_LINEAR)

    img_l_clean= preprocess.cl_correction(img_l_rect)
    img_r_clean = preprocess.cl_correction(img_r_rect)

    # Feature detection

    akaze = cv.AKAZE_create(threshold=akaze_t)
    kpts_l, desc_l = akaze.detectAndCompute(img_l_clean, None)
    kpts_r, desc_r = akaze.detectAndCompute(img_r_clean, None)


    start_time = time.time()
    
    bins_l = {}
    bins_r = {}
    BIN_SIZE = 5
    

    for i, kp in enumerate(kpts_l) :
        bin_nb = int(kp.pt[1]/BIN_SIZE)

        if bin_nb not in bins_l:
            bins_l[bin_nb] = [i]
        else:
            bins_l[bin_nb].append(i)

    for i, kp in enumerate (kpts_r):
        bin_nb = int(kp.pt[1]/BIN_SIZE)

        if bin_nb not in bins_r:
            bins_r[bin_nb] = [i]
        else:
            bins_r[bin_nb].append(i)

    # Les bins sont crées, on peut désormais fait le matching uniquement dans les 2 bins autour
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    pts_g_brut = []
    pts_d_brut = []
    kpts_g_brut = []
    kpts_d_brut = []
    index_g_brut = []

    for keys in bins_l.keys():
        current_keys = bins_l[keys]
        bins_to_research = [keys-1+i for i in range(3) if keys-1+i in bins_r]
        keys_to_research=[]
        for i in bins_to_research:
            keys_to_research += bins_r[i]

        if len(keys_to_research) < 2 or len(current_keys) == 0:
            continue

        knn_matches = matcher.knnMatch(desc_l[current_keys], desc_r[keys_to_research], k=2)


        for match_tuple in knn_matches:
            if len(match_tuple) == 2:
                m, n = match_tuple 
                if m.distance < n.distance * lowe:

                    global_l_index = current_keys[m.queryIdx]
                    global_r_index = keys_to_research[m.trainIdx]

                    index_g_brut.append(global_l_index)

                    pts_g_brut.append(kpts_l[global_l_index].pt)
                    pts_d_brut.append(kpts_r[global_r_index].pt)
                    kpts_g_brut.append(kpts_l[global_l_index])
                    kpts_d_brut.append(kpts_r[global_r_index])


        
    # Matching non optimal qui cherche toute l'image
    '''
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(desc_l, desc_r, k=2)

    print('match done')

    pts_g_brut = []
    pts_d_brut = []
    kpts_g_brut = []
    kpts_d_brut = []

    # Filtrage de Lowe

    for m, n in knn_matches:
        if m.distance < n.distance * lowe:
            pts_g_brut.append(kpts_l[m.queryIdx].pt)
            pts_d_brut.append(kpts_r[m.trainIdx].pt)
            kpts_g_brut.append(kpts_l[m.queryIdx])
            kpts_d_brut.append(kpts_r[m.trainIdx])

    '''

    end_time = time.time()
    duree = end_time - start_time
    print(f"Temps : {duree:.4f} secondes")
    print(f"Matchs trouvés : {len(pts_g_brut)}")

    pts_g_brut = np.float32(pts_g_brut)
    pts_d_brut = np.float32(pts_d_brut)


    inliers_g = []
    inliers_d = []
    pts_g = []
    pts_d = []
    index_g_clean = []

    if len(pts_g_brut) > 10:

        F, mask = cv.findFundamentalMat(pts_g_brut, pts_d_brut, cv.FM_RANSAC, 8.0, 0.99)
        if mask is not None:
            mask = mask.ravel()
            for i in range(len(pts_g_brut)):
                if mask[i] == 1: # Validation RANSAC
                    index_g_clean.append(index_g_brut[i])
                    pts_g.append(pts_g_brut[i])
                    pts_d.append(pts_d_brut[i])
                    inliers_g.append(kpts_g_brut[i])
                    inliers_d.append(kpts_d_brut[i])
                    
            pts_g = np.float32(pts_g)
            pts_d = np.float32(pts_d)

    print('--- Résultats AKAZE ---')
    print(f'# Keypoints Gauche:   \t {len(kpts_l)}')
    print(f'# Keypoints Droite:   \t {len(kpts_r)}')
    print(f'# Matchs valides :\t {len(pts_g)}')


    # Triangulation 

    if len(pts_g) > 10:
        points4D = cv.triangulatePoints(params['P1'], params['P2'], pts_g.T, pts_d.T)
        points3D = (points4D[:3, :] / points4D[3, :]).T


    # Visualisation
   
    vis_g = cv.cvtColor(img_l_clean, cv.COLOR_BGR2RGB)
    vis_d = cv.cvtColor(img_r_clean, cv.COLOR_BGR2RGB)

    cv.drawKeypoints(vis_g, inliers_g, vis_g, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.drawKeypoints(vis_d, inliers_d, vis_d, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    combined = cv.hconcat([vis_g, vis_d])

    plt.figure(figsize=(16, 8))
    plt.title(f"Points utilisés pour la 3D ({len(pts_g)} Inliers)")
    plt.imshow(combined)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    

    # On filtre les points aberrants trop loin (ex: > 15 mètres)

    mask = (points3D[:, 2] > 0.1) & (points3D[:, 2] < 15)
    p3d = points3D[mask]
    index_g_clean = np.array(index_g_clean)
    index_g_clean = index_g_clean[mask]
    pts_g_clean = pts_g[mask]
    desc_l_clean = desc_l[index_g_clean]

    
    ax.scatter(p3d[:, 0], p3d[:, 2], -p3d[:, 1], s=1, c='r') # On inverse Y et Z pour l'affichage
    ax.set_xlabel('X (Largeur)')
    ax.set_ylabel('Z (Profondeur)')
    ax.set_zlabel('Y (Hauteur)')
    plt.title("Aperçu rapide du nuage de points")
    plt.show()
    

    return p3d, pts_g_clean, desc_l_clean 
# La fonction renvoie le nuage 3d, la position X,Y sur l'image gauche des points matchés et leurs decripteurs



if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Pipeline Stéréo 3D avec AKAZE.')
    parser.add_argument('--gauche', help='Chemin image gauche', default='/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/data_sortie_mer/frames/gauche/sortie_left.mp4_fixed/frames/frame_005138.jpg')
    parser.add_argument('--droite', help='Chemin image droite', default='/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/data_sortie_mer/frames/droite/sortie_right.mp4_fixed/frames/frame_005138.jpg')
    parser.add_argument("--akaze", type=float, default=0.00001, help="Seuil de détection AKAZE")
    parser.add_argument('--lowe', type=float, default=0.8, help='Ratio de Lowe')
    parser.add_argument('--ytol', type=float, default=10.0, help='Tolérance horizontale (pixels)')
    args = parser.parse_args()
    generate_cloud(args.gauche, args.droite)

