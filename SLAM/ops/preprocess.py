import cv2
import numpy as np
 
#image_path = "C:/Users/ssabb/Downloads/IMG_5477.mp4"  # Change to your input video
'''
frame_path = "SLAM/image6.jpg"
frame = cv2.imread(frame_path)
'''

def cl_vert(image):
    return image[:,:,1]

def cl_correction(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminescence = lab_image[:,:,0]
    lumi_equalized = cv2.equalizeHist(luminescence)
    lab_image[:,:,0] = lumi_equalized
    bgr = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    bgr[:,:,2] = 50 + 0.75*bgr[:,:,2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray

# cv2.namedWindow('Channel vert', cv2.WINDOW_KEEPRATIO)
# cv2.namedWindow('Correction doc', cv2.WINDOW_KEEPRATIO)
# cv2.namedWindow('Greyscale', cv2.WINDOW_KEEPRATIO)
# cv2.imshow("Channel vert",cl_vert(frame))
# cv2.imshow("Correction doc", cl_correction(frame))
# cv2.imshow("Greyscale", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
# cv2.waitKey()
# cv2.destroyAllWindows()
