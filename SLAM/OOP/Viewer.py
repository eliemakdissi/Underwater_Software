import open3d as o3d
import numpy as np
import time

class Viewer:
    def __init__(self):
        print("[VIEWER] Initialisation de la fenêtre 3D (Open3D)...")
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="SLAM 3D Live - G2O", width=1280, height=720)
        
        # Objet : Nuage de points
        self.pcd = o3d.geometry.PointCloud()
        
        # Objet : Trajectoire (lignes)
        self.trajectory = o3d.geometry.LineSet()
        
        # Ajout à la scène
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.trajectory)
        
        self.is_initialized = False
        self.frame_counter = 0

    def update(self, map_obj):
        """ 
        Met à jour la géométrie 3D de manière non bloquante.
        Appelé depuis la boucle principale.
        """
        self.frame_counter += 1

        # On saute 4 frames sur 5 pour garder le système fluide
        if self.frame_counter % 5 == 0:
            
            # 1. Extraction rapide des données sous Mutex
            with map_obj.data_mutex_:
                points = map_obj.get_all_map_points()
                keyframes = map_obj.get_all_keyframes()

            # 2. Mise à jour de la carte (si elle n'est pas vide)
            if points and keyframes:
                # --- POINTS 3D ---
                xyz = np.array([mp.pos for mp in points])
                # Filtre de profondeur pour éviter les points aberrants qui dézooment tout
                mask = (xyz[:, 2] > 0.1) & (xyz[:, 2] < 20.0)
                xyz_clean = xyz[mask]
                
                self.pcd.points = o3d.utility.Vector3dVector(xyz_clean)
                self.pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([1.0, 0.0, 0.0]), (len(xyz_clean), 1)))

                # --- TRAJECTOIRE ---
                kfs_sorted = sorted(keyframes, key=lambda x: x.id_)
                traj_pts = [np.linalg.inv(kf.pose)[:3, 3] for kf in kfs_sorted]
                
                self.trajectory.points = o3d.utility.Vector3dVector(np.array(traj_pts))

                if len(traj_pts) > 1:
                    lines = [[i, i+1] for i in range(len(traj_pts)-1)]
                    self.trajectory.lines = o3d.utility.Vector2iVector(lines)
                    self.trajectory.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.0, 0.5, 1.0]), (len(lines), 1))) # Bleu clair

                self.vis.update_geometry(self.pcd)
                self.vis.update_geometry(self.trajectory)

                # Centrer la caméra lors de la première vraie frame
                if not self.is_initialized and len(xyz_clean) > 0:
                    self.vis.reset_view_point(True)
                    self.is_initialized = True

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        """ Garde la fenêtre ouverte à la fin de la vidéo jusqu'à fermeture manuelle """
        print("[VIEWER] Fin de la séquence. Appuyez sur Q ou fermez la fenêtre Open3D.")
        self.vis.run()
        self.vis.destroy_window()