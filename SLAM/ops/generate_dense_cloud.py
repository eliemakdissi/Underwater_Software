import preprocess 
import numpy as np
import cv2 as cv
import pickle
import time
import argparse
import matplotlib.pyplot as plt


def generate_dense_cloud(PATH_IMG_L: str, PATH_IMG_R: str, params, pose_camera_globale):

    img_l_brute = cv.imread(PATH_IMG_L)
    img_r_brute = cv.imread(PATH_IMG_R)
    IMAGE_SIZE = (img_l_brute.shape[1], img_l_brute.shape[0])

    # Preprocessing
    mapl_x, mapl_y = cv.initUndistortRectifyMap(params['mtx1'], params['dist1'], params['R1'], params['P1'], IMAGE_SIZE, cv.CV_32FC1)
    mapr_x, mapr_y = cv.initUndistortRectifyMap(params['mtx2'], params['dist2'], params['R2'], params['P2'], IMAGE_SIZE, cv.CV_32FC1)

    img_l_rect = cv.remap(img_l_brute, mapl_x, mapl_y, cv.INTER_LINEAR)
    img_r_rect = cv.remap(img_r_brute, mapr_x, mapr_y, cv.INTER_LINEAR)

    img_l_clean = preprocess.cl_correction(img_l_rect)
    img_r_clean = preprocess.cl_correction(img_r_rect)

    # StereoSGBM - feature detection

    scale = 0.5
    img_l_half = cv.resize(img_l_clean, None, fx=scale, fy=scale)
    img_r_half = cv.resize(img_r_clean, None, fx=scale, fy=scale)

    # 2. Configuration SGBM (Ajustée pour la petite image)
    block_size = 7  # Taille du bloc sur la petite image
    num_disp = 16 * 10 # 160 pixels de recherche sur l'image réduite (équivaut à 320 en 1080p)
    
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10, 
        speckleWindowSize=150, # Nettoie les points isolés (les confettis)
        speckleRange=2,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 3. Calcul sur les petites images
    disp_half = stereo.compute(img_l_half, img_r_half).astype(np.float32) / 16.0

    # 4. On remet la disparité à la taille d'origine (1080p) pour utiliser la matrice Q
    disparity = cv.resize(disp_half, (img_l_clean.shape[1], img_l_clean.shape[0]), interpolation=cv.INTER_NEAREST)
    disparity = disparity / scale # On n'oublie pas de multiplier les valeurs par 2 !

    '''
    plt.figure(figsize=(10, 5))
    plt.imshow(disparity, cmap='jet', vmin=0, vmax=num_disp)
    plt.colorbar(label='Disparity')
    plt.title("GBM")
    plt.show()
    '''

    # 3D local
    Q = params['Q']
    points_3D_local = cv.reprojectImageTo3D(disparity, Q)

    # Filtrage
    mask = (disparity > 0) & (points_3D_local[:, :, 2] < 5.0) & (points_3D_local[:, :, 2] > 0.1)
    points_locaux_filtres = points_3D_local[mask]
    couleurs_filtrees = cv.cvtColor(img_l_rect, cv.COLOR_BGR2RGB)[mask] / 255.0

    # Cooordonées globales pour PnP
    points_homogenes = np.hstack((points_locaux_filtres, np.ones((len(points_locaux_filtres), 1))))
    points_globaux = (pose_camera_globale @ points_homogenes.T).T[:, :3]

    return points_globaux, couleurs_filtrees