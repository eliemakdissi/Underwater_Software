# Implementation of the Map class
import open3d as o3d
import threading
import time
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

import Params

class Map:

    def __init__(self):

        self.landmarks_ = {}
        self.active_landmarks_ = {}
        self.keyframes_ = {}
        self.active_keyframes_ = {}

        self.current_frame_ = None

        self.data_mutex_ = threading.Lock()

    def insert_keyframes(self, frame):

        with self.data_mutex_:
            self.current_frame_ = frame
            self.keyframes_[frame.id_] = frame
            self.active_keyframes_[frame.id_] = frame

        if len(self.active_keyframes_) > Params.get('num_active_keyframes'):
            self.remove_old_keyframe()

    def insert_map_point(self, map_point):

        with self.data_mutex_:
            self.landmarks_[map_point.id_] = map_point
            self.active_landmarks_[map_point.id_] = map_point

    def get_all_map_points(self):
        with self.data_mutex_:
            return list(self.landmarks_.values())
        
    def get_all_keyframes(self):
        with self.data_mutex_:
            return list(self.keyframes_.values())
        
    def get_active_map_points(self):
        with self.data_mutex_:
            return list(self.active_landmarks_.values())
        
    def get_active_keyframes(self):
        with self.data_mutex_:
            return list(self.active_keyframes_.values())
        

    def clean_map(self):
        
        with self.data_mutex_:
            ids_to_remove = []

            for mp_id, map_point in self.landmarks_.items() : 
                if map_point.is_outlier_ or map_point.observed_times_ <= 0:
                    ids_to_remove.append(mp_id)

            for mp_id in ids_to_remove:
                self.landmarks_.pop(mp_id, None)
                self.active_landmarks_.pop(mp_id, None)


    def remove_old_keyframe(self):

        limit = Params.get('num_active_keyframes')
        while len(self.active_keyframes_) > limit:
            min_id = min(self.active_keyframes_.keys())
            self.active_keyframes_.pop(min_id, None)

    # ------------------------------------------------------------------
    # Dash live 3-D plot
    # ------------------------------------------------------------------

    def start_dash_server(self):
        self.dash_app = Dash(__name__)
        self.dash_app.layout = html.Div([
            html.H2("SLAM - Live 3D"),
            dcc.Graph(id='slam-3d', style={'height': '85vh'}),
            dcc.Interval(id='interval', interval=2000, n_intervals=0)
        ])

        @self.dash_app.callback(
            Output('slam-3d', 'figure'),
            Input('interval', 'n_intervals')
        )
        def update_plot(_):
            return self._build_figure()

        dash_thread = threading.Thread(
            target=self.dash_app.run,
            kwargs={'host': '0.0.0.0', 'port': 8050, 'debug': False},
            daemon=True
        )
        dash_thread.start()
        print("[Map] Dash server started at http://0.0.0.0:8050")

    def _build_figure(self):
        keyframes = self.get_all_keyframes()
        all_points = self.get_all_map_points()

        # --- Trajectory trace (always present, empty if no data) ---
        traj_x, traj_y, traj_z = [], [], []
        start_x, start_y, start_z = [], [], []

        if keyframes:
            cam_pos = []
            for kf in keyframes:
                T_wc = np.linalg.inv(kf.pose_)
                cam_pos.append(T_wc[:3, 3])
            cam_pos = np.array(cam_pos)
            traj_x, traj_y, traj_z = cam_pos[:, 0], cam_pos[:, 2], cam_pos[:, 1]
            start_x, start_y, start_z = [cam_pos[0, 0]], [cam_pos[0, 2]], [cam_pos[0, 1]]

        # --- Points trace (always present, empty if no data) ---
        pts_x, pts_y, pts_z = [], [], []
        pts_colors = []
        n_pts_filtered = 0

        if all_points:
            valid_mps = [mp for mp in all_points if not mp.is_outlier_]
            if valid_mps:
                pts = np.array([mp.pos_ for mp in valid_mps])
                cols = np.array([mp.color_ for mp in valid_mps], dtype=np.int32)
                mask = (pts[:, 2] > 0.05) & (pts[:, 2] < 20.0)
                pts = pts[mask]
                cols = cols[mask]
                if len(pts) > 0:
                    pts_x, pts_y, pts_z = pts[:, 0], pts[:, 2], pts[:, 1]
                    pts_colors = [f'rgb({r},{g},{b})' for r, g, b in cols]
                    n_pts_filtered = len(pts)

        n_kf = len(keyframes) if keyframes else 0
        n_pts = len(all_points) if all_points else 0

        # Always return exactly 3 traces in the same order
        fig = go.Figure(data=[
            go.Scatter3d(
                x=traj_x, y=traj_y, z=traj_z,
                mode='lines+markers',
                marker=dict(size=4, color='blue'),
                line=dict(color='blue', width=3),
                name='Trajectory'
            ),
            go.Scatter3d(
                x=start_x, y=start_y, z=start_z,
                mode='markers',
                marker=dict(size=8, color='green'),
                name='Start'
            ),
            go.Scatter3d(
                x=pts_x, y=pts_y, z=pts_z,
                mode='markers',
                marker=dict(size=2, color=pts_colors if pts_colors else 'red', opacity=0.9),
                name=f'Points ({n_pts_filtered})'
            ),
        ])

        fig.update_layout(
            title=f'SLAM Live - {n_kf} KFs, {n_pts} pts',
            scene=dict(
                xaxis=dict(title='X', range=[-5, 5]),
                yaxis=dict(title='Z (depth)', range=[-5, 5]),
                zaxis=dict(title='Y', range=[-5, 5]),
                aspectmode='cube',
                dragmode='orbit',
                uirevision='slam-scene',
            ),
            uirevision='slam',
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig
    def build_mesh(self, method="alpha"):
        # Convert all map points to an Open3D PointCloud object.
        all_points = self.get_all_map_points()

        pts = np.array([mp.pos_ for mp in all_points if not mp.is_outlier_])
        if len(pts) == 0:
            return o3d.geometry.PointCloud()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=10)
        start= time.time()
        if method== "pointball":
            radii = [0.005, 0.01, 0.02, 0.04,0.1]
            rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
        elif method=="alpha":
            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
            for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
                print(f"alpha={alpha:.3f}")
                rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha, tetra_mesh, pt_map)
                rec_mesh.compute_vertex_normals()
                o3d.visualization.draw_geometries([pcd, rec_mesh],mesh_show_back_face=True)
        else:
            with o3d.utility.VerbosityContextManager(
                    o3d.utility.VerbosityLevel.Debug) as cm:
                for alpha in range(10, 50, 20):
                    print(f"alpha={alpha:.3f}")
                    rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd, depth=alpha)
                    rec_mesh.compute_vertex_normals()
                    o3d.visualization.draw_geometries([pcd, rec_mesh],mesh_show_back_face=True)
        end = time.time()
        print("temps de meshing : ", end-start)
        o3d.visualization.draw_geometries([pcd, rec_mesh],mesh_show_back_face=True)
        return pcd, rec_mesh
        
