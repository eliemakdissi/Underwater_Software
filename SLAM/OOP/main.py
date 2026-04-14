import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from Frame import Frame
from Map import Map
from Frontend import Frontend
from Backend2 import Backend

def main():
    print("--- Initialisation du SLAM ---")

    # 1. Initialisation de l'architecture
    slam_map = Map()
    # Assure-toi que Frame.params contient bien tes paramètres de calibration stéréo
    frontend = Frontend(params_stereo_=Frame.params, map=slam_map)
    backend = Backend(params_stereo=Frame.params, slam_map=slam_map)

    # Variables pour le visuel final
    trajectoire_camera = []

    # 2. Boucle Principale
    # Ajuste le range() selon le nombre d'images de ton dataset
    start_frame = 2
    end_frame = 20 
    
    for i in range(start_frame, end_frame):
        t_start = time.time()
        
        # Ajuste les chemins vers tes images de test
        path_l = f'SLAM/images_test/set_3_caillou/frame_{i:04d}_l.jpg'
        path_r = f'SLAM/images_test/set_3_caillou/frame_{i:04d}_r.jpg'
        
        if not os.path.exists(path_l) or not os.path.exists(path_r):
             print(f"⚠️ Image {i} introuvable, passage à la suivante.")
             continue
        
        img_l = cv.imread(path_l, cv.IMREAD_COLOR) 
        img_r = cv.imread(path_r, cv.IMREAD_COLOR)

        # Création et envoi au Frontend
        current_frame = Frame.create_frame(time_stamp=float(i), left_img=img_l, right_img=img_r)
        success = frontend.add_frame(current_frame)

        # ==========================================
        # DEBUG VISUEL : Temps Réel (OpenCV)
        # ==========================================
        if success:
            # Extraction de la position globale de la caméra
            # T_wc (World -> Camera) = inv(pose). La translation est dans [:3, 3]
            pose_inv = np.linalg.inv(current_frame.pose_)
            trajectoire_camera.append(pose_inv[:3, 3])

            # Dessin de l'image gauche avec les inliers
            vis_img = cv.cvtColor(current_frame.clean_left_img_, cv.COLOR_GRAY2BGR)
            inliers_count = 0
            
            for feat in current_frame.features_left_:
                if feat.map_point_ is not None:
                    # Point lié à la 3D = Inlier valide
                    x, y = int(feat.position_.pt[0]), int(feat.position_.pt[1])
                    cv.circle(vis_img, (x, y), 3, (0, 255, 0), -1)
                    inliers_count += 1
            
            # Affichage du statut sur l'image
            status_text = f"Frame: {i} | Inliers: {inliers_count} | KFs: {len(slam_map.get_all_keyframes())}"
            cv.putText(vis_img, status_text, (20, 40), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv.imshow("SLAM Tracking", vis_img)
            cv.waitKey(1)

            # Backend optimisation + live plot
            backend.process_new_keyframes()
            backend.plot_live()

        else:
            print(f"Echec du tracking sur la frame {i}")

        print(f"Frame {i} traitee en {(time.time() - t_start)*1000:.1f} ms")

    cv.destroyAllWindows()

    # ==========================================
    # Final optimised 3D plot (blocking)
    # ==========================================
    print("\n--- Carte 3D optimisee ---")
    plt.ioff()
    backend.plot_live()          # one last refresh with final state
    plt.ioff()                   # switch to blocking mode
    plt.show()                   # keeps the window open until closed

if __name__ == '__main__':
    main()