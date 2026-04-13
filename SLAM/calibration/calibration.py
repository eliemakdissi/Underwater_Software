import numpy as np
import cv2
import glob
import pickle
import g2o

"""ATTENTION : L'implémentation en fisheye n'a pas été terminée car jugée inutile, donc elle ne marche à priori pas."""

def ini_calib(fisheye=0):
    """permet de calculer les paramètres de la caméra à partir d'un set de photos de calibration"""
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('SLAM/calibration/set_doublecam/*r.jpg')
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
            cv2.imshow('img', img)
            cv2.waitKey(0)
        else:
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            pass
    print("nb d'images non reconnues : ", len(images)-i)
    cv2.destroyAllWindows()
    h,  w = (1080,1920)
    if fisheye ==1:
        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(np.reshape(objpoints,(np.shape(objpoints)[0],np.shape(objpoints)[1],1,np.shape(objpoints)[2])), imgpoints, gray.shape[::-1], None, None)
        newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, (w,h), np.eye(3), (w,h))
    else:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

    print("erreur de reprojection ?:", ret)
    return [mtx, dist, newcameramtx, roi]

def cor_calib(img,mtx,dist,newcameramtx, roi,fisheye=0, crop=1):
    """corrige la distorsion et refait l'échelle de l'image"""
    h,  w = img.shape[:2]
    # undistort
    if fisheye==1:
        dst = cv2.fisheye.undistortImage(img, mtx, dist, None, newcameramtx)
    else:
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    if crop ==1:
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
    return dst

def test_calib(mtx, dist, newcameramtx, roi):
    """permet de comparer l'image de base et l'image corrigée"""
    img = cv2.imread('SLAM/calibration/controle/controle2.jpg')
    cv2.namedWindow('image corrigee', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('image originale', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('image originale',img)
    cv2.imshow('image corrigee',cor_calib(img,mtx,dist, newcameramtx, roi))
    cv2.waitKey()
    cv2.imwrite('SLAM/calibration//controle/controle2calibeau.jpg',cor_calib(img,mtx,dist,newcameramtx, roi))

# Sauvegarde de nouveaux paramètres calculés
# with open("SLAM/calibration/param/parametres_calibeau_premierecam.txt", 'wb') as f
#     l= ini_calib()
#     print(l)
#     pickle.dump(l, f)

# Calcul de nouveaux paramètres, test
mtx, dist, newcameramtx, roi = ini_calib()
print(mtx, dist, newcameramtx, roi)
test_calib(mtx, dist, newcameramtx, roi)

# Récupération de paramètres
# with open("SLAM/calibration/parametres_calib2.txt", 'rb') as f:
#     param = pickle.load(f)
 