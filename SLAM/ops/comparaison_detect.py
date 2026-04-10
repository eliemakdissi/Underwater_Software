import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt

def compare_detectors(image_path):
    img_color = cv.imread(image_path)
    if img_color is None:
        print(f"Erreur : Impossible de charger l'image {image_path}")
        return
    
    canal_vert = img_color[:, :, 1]
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_gray = clahe.apply(canal_vert)

    # Détecteurs
    sift_01 = cv.SIFT_create(contrastThreshold=0.01, edgeThreshold=10, nOctaveLayers=4)
    sift_02 = cv.SIFT_create(contrastThreshold=0.02, edgeThreshold=10, nOctaveLayers=4)
    sift_05 = cv.SIFT_create(contrastThreshold=0.05, edgeThreshold=10, nOctaveLayers=4)
    
    orb_5 = cv.ORB_create(nfeatures=5000, fastThreshold=10, scaleFactor=1.1, patchSize=41, edgeThreshold=41)
    orb_10 = cv.ORB_create(nfeatures=10000, fastThreshold=10, scaleFactor=1.1, patchSize=41, edgeThreshold=41)
    # Correction ici : orb_15 mis à 15000 au lieu de 5000
    orb_15 = cv.ORB_create(nfeatures=15000, fastThreshold=10, scaleFactor=1.1, patchSize=41, edgeThreshold=41)
    
    akaze_weickert = cv.AKAZE_create(threshold=0.0005, nOctaves=4, nOctaveLayers=4, diffusivity=cv.KAZE_DIFF_WEICKERT)
    akaze_charbonnier = cv.AKAZE_create(threshold=0.0005, nOctaves=4, nOctaveLayers=4, diffusivity=cv.KAZE_DIFF_CHARBONNIER)
    akaze_classic = cv.AKAZE_create(threshold=0.0005, nOctaves=4, nOctaveLayers=4)

    detectors = {
        "SIFT_01 ": sift_01,
        "SIFT_02" : sift_02,
        "SIFT_05" : sift_05,
        "ORB_5 ": orb_5,
        "ORB_10": orb_10,
        "ORB_15": orb_15,
        "AKAZE_WEICKERT": akaze_weickert,
        "AKAZE_CHARBONNIER": akaze_charbonnier,
        "AKAZE ": akaze_classic
    }

    results = {}

    # Exécution et mesure du temps
    for name, detector in detectors.items():
        start_time = time.time()

        keypoints = detector.detect(img_gray, None)
        duration = (time.time() - start_time) * 1000 # en millisecondes
        
        # Le flag DRAW_RICH_KEYPOINTS dessine des cercles proportionnels à la taille de la feature
        img_drawn = cv.drawKeypoints(img_gray, keypoints, None, color=(0, 255, 0), 
                                     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        results[name] = {
            "image": img_drawn,
            "count": len(keypoints),
            "time": duration
        }
        
        print(f"{name:20s} : {len(keypoints):5d} points trouvés en {duration:6.1f} ms")

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle("Comparaison des Détecteurs (Spécial Sous-Marin)", fontsize=16)

    for ax, (name, data) in zip(axes.ravel(), results.items()):
        ax.imshow(data["image"])
        ax.set_title(f"{name}\n{data['count']} points | {data['time']:.1f} ms")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    CHEMIN_IMAGE = 'SLAM/images_test/set_3_caillou/frame_0002_l.jpg'
    compare_detectors(CHEMIN_IMAGE)