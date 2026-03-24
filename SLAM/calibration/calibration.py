import numpy as np
import cv2
import glob
import pickle

def ini_calib():
    """permet de calculer les paramètres de la caméra à partir d'un set de photos de calibration"""
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('SLAM/calibration/set_2/*.jpg')
    i=0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,7), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            i +=1
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,7), corners2, ret)
    print("nb d'images non reconnues : ", len(images)-i)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    img = cv2.imread('SLAM/calibration/controle.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    return [mtx, dist, newcameramtx, roi]

def cor_calib(img,mtx,dist,newcameramtx, roi):
    """corrige la distorsion et refait l'échelle de l'image"""
    h,  w = img.shape[:2]

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def test_calib(mtx, dist, newcameramtx, roi):
    """permet de comparer l'image de base et l'image corrigée"""
    img = cv2.imread('SLAM/calibration/controle.jpg')
    cv2.namedWindow('image corrigee', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('image originale', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('image originale',img)
    cv2.imshow('image corrigee',cor_calib(img,mtx,dist, newcameramtx, roi))
    cv2.waitKey()
    cv2.imwrite('SLAM/calibration/controle_corrigee_c2.jpg',cor_calib(img,mtx,dist,newcameramtx, roi))

# Sauvegarde des paramètres calculés
# with open("SLAM/calibration/parametres_calib2.txt", 'wb') as f:
#     l= ini_calib()
#     print(l)
#     pickle.dump(l, f)

# Calcul de nouveaux paramètres
# mtx, dist, newcameramtx, roi = l
# test_calib(mtx, dist, newcameramtx, roi)

# Récupération de paramètres
# with open("SLAM/calibration/parametres_calib2.txt", 'rb') as f:
#     param = pickle.load(f)
#     print(param)