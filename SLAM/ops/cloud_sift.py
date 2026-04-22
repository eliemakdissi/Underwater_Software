# Generate a 3D cloud of points using sift

import ops.preprocess 
import numpy as np
import cv2 as cv
import pickle
import time
import argparse
import matplotlib.pyplot as plt
import os
import open3d as o3d

DOSSIER_SORTIE = "SLAM/test_sift_ply"
os.makedirs(DOSSIER_SORTIE, exist_ok=True)



def save_point_cloud_ply(filename, xyz_points):
    """
    Sauvegarde un tableau numpy de points 3D au format .ply ASCII.
    """
    print(f"Sauvegarde du nuage de points dans {filename}...")
    with open(filename, 'w') as f:
        # En-tête strict du format PLY
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        # Écriture des coordonnées
        for p in xyz_points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
    print("Sauvegarde terminée !")


def generate_cloud(PATH_IMG_L : str, PATH_IMG_R : str, akaze_t : float = 0.0001, lowe : float = 0.95) :

    img_l_brute = cv.imread(PATH_IMG_L) 
    img_r_brute = cv.imread(PATH_IMG_R)
    
    h, w = img_l_brute.shape[:2]
    IMAGE_SIZE = (w, h)

    print(IMAGE_SIZE)

    """ with open('/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/calibration/param/stereo_a_lenvers.pkl', 'rb') as f:
        params = pickle.load(f) """

    params = {'mtx1': np.array([[1.25051139e+03, 0.00000000e+00, 9.83089787e+02],
       [0.00000000e+00, 1.25043408e+03, 5.31404956e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), 'dist1': np.array([[-0.32465656,  0.19275159, -0.00075876,  0.0038447 , -0.0540854 ]]), 'mtx2': np.array([[1.24269700e+03, 0.00000000e+00, 9.67192665e+02],
       [0.00000000e+00, 1.24336262e+03, 5.71800196e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), 'dist2': np.array([[-0.32106801,  0.18508454,  0.00183416,  0.0011332 , -0.04652458]]), 'R': np.array([[ 0.99926041, -0.00330228,  0.03831103],
       [ 0.00421827,  0.99970658, -0.02385302],
       [-0.03822102,  0.02399699,  0.99898113]]), 'T': np.array([[-0.04065602],
       [-0.0002887 ],
       [-0.00069275]]), 'R1': np.array([[ 0.998469  ,  0.00420484,  0.05515408],
       [-0.00353962,  0.99991988, -0.01215338],
       [-0.05520077,  0.01193954,  0.99840389]]), 'R2': np.array([[ 0.99982966,  0.00709983,  0.01703628],
       [-0.00730462,  0.99990144,  0.01198936],
       [-0.01694948, -0.01211176,  0.99978299]]), 'P1': np.array([[1.19671761e+03, 0.00000000e+00, 9.14563248e+02, 0.00000000e+00],
       [0.00000000e+00, 1.19671761e+03, 5.57418053e+02, 0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]]), 'P2': np.array([[ 1.19671761e+03,  0.00000000e+00,  9.14563248e+02,
        -4.86620583e+01],
       [ 0.00000000e+00,  1.19671761e+03,  5.57418053e+02,
         0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
         0.00000000e+00]]), 'roi1': (0, 0, 1920, 1080), 'roi2': (0, 0, 1920, 1080), 'Q': np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -9.14563248e+02],
       [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        -5.57418053e+02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.19671761e+03],
       [ 0.00000000e+00,  0.00000000e+00,  2.45924166e+01,
        -0.00000000e+00]])}

    #print(params)

    # Preprocessing

    mapl_x, mapl_y = cv.initUndistortRectifyMap(params['mtx1'], params['dist1'], params['R1'], params['P1'], IMAGE_SIZE, cv.CV_32FC1)
    mapr_x, mapr_y = cv.initUndistortRectifyMap(params['mtx2'], params['dist2'], params['R2'], params['P2'], IMAGE_SIZE, cv.CV_32FC1)

    img_l_rect = cv.remap(img_l_brute, mapl_x, mapl_y, cv.INTER_LINEAR)
    img_r_rect = cv.remap(img_r_brute, mapr_x, mapr_y, cv.INTER_LINEAR)

    # Debug lignes épipolaires
    
    vis_g = cv.resize(img_l_rect, (640, 480)) # Redimensionner pour affichage
    vis_d = cv.resize(img_r_rect, (640, 480))
    combined = cv.hconcat([vis_g, vis_d])
    
    # Dessiner des lignes horizontales vertes tous les 30 pixels
    for y in range(0, combined.shape[0], 30):
        cv.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)
        
    """ cv.imshow("Test Epipolaire", combined)
    cv.waitKey(0)
    cv.destroyAllWindows() """
    

    img_l_clean = ops.preprocess.cl_vert(img_l_rect)
    img_r_clean = ops.preprocess.cl_vert(img_r_rect)

    #img_l_clean= preprocess.cl_correction(img_l_rect)
    #img_r_clean = preprocess.cl_correction(img_r_rect)

    # Feature detection

    sift = cv.SIFT_create(contrastThreshold=0.01, edgeThreshold=10, nOctaveLayers=4)
    kpts_l, desc_l = sift.detectAndCompute(img_l_clean, None)
    kpts_r, desc_r = sift.detectAndCompute(img_r_clean, None)


    start_time = time.time()
    
    bins_l = {}
    bins_r = {}
    BIN_SIZE = 25
    

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
    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

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
    matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
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

    pts_g_brut = np.array(pts_g_brut, dtype=float)
    pts_d_brut = np.array(pts_d_brut, dtype=float)


    inliers_g = []
    inliers_d = []
    pts_g = []
    pts_d = []
    index_g_clean = []
    
    if len(pts_g_brut) > 10:
        diff_y = np.abs(pts_g_brut[:, 1] - pts_d_brut[:, 1])
        disparite = pts_g_brut[:, 0] - pts_d_brut[:, 0]

        EPIPOLAR_CONSTRAINT = 5.0
        DISPARITY_CONSTRAINT = 0.2
        good_mask = (diff_y < EPIPOLAR_CONSTRAINT) & (disparite>DISPARITY_CONSTRAINT) # Filtre des matchs 

        pts_g = pts_g_brut[good_mask]
        pts_d = pts_d_brut[good_mask]

        # Reconstruire index_g_clean et les keypoints pour visualisation
        index_g_brut_arr = np.array(index_g_brut)
        index_g_clean = index_g_brut_arr[good_mask]

        kpts_g_brut_arr = np.array(kpts_g_brut, dtype=object)
        kpts_d_brut_arr = np.array(kpts_d_brut, dtype=object)
        inliers_g = kpts_g_brut_arr[good_mask].tolist()
        inliers_d = kpts_d_brut_arr[good_mask].tolist()

        pts_g = np.array(pts_g, dtype=float)
        pts_d = np.array(pts_d, dtype=float)
    
    
    print('--- Résultats SIFT ---')
    print(f'# Keypoints Gauche:   \t {len(kpts_l)}')
    print(f'# Keypoints Droite:   \t {len(kpts_r)}')
    print(f'# Matchs bruts:       \t {len(pts_g_brut)}')
    print(f'# Matchs valides :\t {len(pts_g)}')


    '''
    points3D=[]


    if len(pts_g) > 10:

        for i in range(len(pts_g)):
            if np.abs(pts_g[i][1] - pts_d[i][1]) > 10: continue

            points4D = np.array([pts_g[i][0], pts_g[i][1], pts_g[i][0]-pts_d[i][0], 1]).T
            #print(points4D, pts_g[i][0], pts_g[i][1], pts_d[i][0], pts_d[i][1])
            points4D = params['Q'] @ points4D
            points3D.append((points4D[:3] / points4D[3]).T)

    '''
    
    if len(pts_g) > 10:
        points4D = cv.triangulatePoints(params['P1'], params['P2'], pts_g.T, pts_d.T)
        points3D = (points4D[:3, :] / points4D[3, :]).T

    # Visualisation
    vis_g = cv.cvtColor(img_l_clean, cv.COLOR_BGR2RGB)
    vis_d = cv.cvtColor(img_r_clean, cv.COLOR_BGR2RGB)

    cv.drawKeypoints(vis_g, inliers_g, vis_g, color=(255, 0, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.drawKeypoints(vis_d, inliers_d, vis_d, color=(255, 0, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    combined = cv.hconcat([vis_g, vis_d])
    
    """ plt.figure(figsize=(16, 8))
    plt.title(f"Points utilisés pour la 3D ({len(pts_g)} Inliers)")
    plt.imshow(combined)
    plt.axis('off')
    plt.tight_layout()
    plt.show() """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    p3d = []
    pts_g_clean_final = []
    desc_l_clean_final = []

    # On filtre les points aberrants trop loin (ex: > 5 mètres)
    if len(points3D) > 0:
        mask = (points3D[:, 2] > 0.01) & (points3D[:, 2] < 5.0)
        p3d = points3D[mask]
        
        save_point_cloud_ply('test_sift.ply', p3d)

        # Les masques refonctionnent car les tailles sont identiques
        index_g_clean = np.array(index_g_clean)[mask]
        pts_g_clean_final = pts_g[mask]
        desc_l_clean_final = desc_l[index_g_clean]
        
        """ ax.scatter(p3d[:, 0], p3d[:, 2], -p3d[:, 1], s=1, c='r') # On inverse Y et Z pour l'affichage
        ax.set_xlabel('X (Largeur)')
        ax.set_ylabel('Z (Profondeur)')
        ax.set_zlabel('Y (Hauteur)')
        plt.title(f"Aperçu rapide du nuage de points - {len(points3D)} bruts - {len(p3d)} points valides")
        plt.show() """
    
    return p3d, pts_g_clean_final, desc_l_clean_final 

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Pipeline Stéréo 3D avec SIFT.')
    parser.add_argument('--gauche', help='Chemin image gauche', default='SLAM/images_test/image_caillou/frame_0001_l.jpg')
    parser.add_argument('--droite', help='Chemin image droite', default='SLAM/images_test/image_caillou/frame_0001_r.jpg')
    parser.add_argument("--akaze", type=float, default=0.001, help="Seuil de détection AKAZE")
    parser.add_argument('--lowe', type=float, default=0.8, help='Ratio de Lowe')
    parser.add_argument('--ytol', type=float, default=10.0, help='Tolérance horizontale (pixels)')
    args = parser.parse_args()
    generate_cloud(args.gauche, args.droite)
