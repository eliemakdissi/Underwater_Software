import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from Frame import Frame
from Map import Map
from Frontend import Frontend

def main():
    print("--- Initialisation du SLAM ---")
    
    # 1. Initialisation de l'architecture
    slam_map = Map()
    # Assure-toi que Frame.params contient bien tes paramètres de calibration stéréo
    frontend = Frontend(params_stereo_=Frame.params, map=slam_map)

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
            cv.waitKey(1) # Laisse l'image affichée 1ms
            
        else:
            print(f"❌ Échec du tracking sur la frame {i}")

        print(f"Frame {i} traitée en {(time.time() - t_start)*1000:.1f} ms")

    cv.destroyAllWindows()

    # ==========================================
    # DEBUG VISUEL : Fin de run (Matplotlib 3D)
    # ==========================================
    print("\n--- Génération de la Carte 3D ---")
    
    tous_les_points = slam_map.get_all_map_points()
    if len(tous_les_points) == 0:
        print("La carte est vide !")
        return

    # Extraction des coordonnées X,Y,Z
    xyz_global = np.array([mp.pos_ for mp in tous_les_points])
    trajectoire = np.array(trajectoire_camera)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Filtre les points aberrants pour l'affichage (ex: Z > 20m ou Z < 0)
    mask = (xyz_global[:, 2] > 0.05) & (xyz_global[:, 2] < 20.0)
    p3d = xyz_global[mask]

    # Nuage de points (Rouge)
    ax.scatter(p3d[:, 0], p3d[:, 2], -p3d[:, 1], s=1, c='r', alpha=0.5, label="MapPoints")
    
    # Trajectoire de la caméra (Bleu)
    if len(trajectoire) > 0:
        ax.plot(trajectoire[:, 0], trajectoire[:, 2], -trajectoire[:, 1], 
                c='b', linewidth=2, marker='o', markersize=4, label="Trajectoire Caméra")
        
        # Marquer le point de départ en vert
        ax.scatter(trajectoire[0, 0], trajectoire[0, 2], -trajectoire[0, 1], c='g', s=50, label="Départ", zorder=5)

    ax.set_xlabel('X (Droite/Gauche)')
    ax.set_ylabel('Z (Profondeur)')
    ax.set_zlabel('Y (Haut/Bas)')
    
    # Ajuster le ratio des axes pour une vue plus réaliste
    ax.set_box_aspect([1, 1, 1]) 
    
    plt.title(f"Carte 3D du SLAM ({len(p3d)} points, {len(slam_map.get_all_keyframes())} Keyframes)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()