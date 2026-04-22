# Implementation of the Map class
import base64
import json
import os
import open3d as o3d
import threading
import time
import cv2 as cv
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

import Params

VIEW_FILE = '.slam_view.json'


def _load_saved_camera():
    if not os.path.exists(VIEW_FILE):
        return None
    try:
        with open(VIEW_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

class Map:

    def __init__(self):

        self.landmarks_ = {}
        self.active_landmarks_ = {}
        self.keyframes_ = {}
        self.active_keyframes_ = {}

        self.current_frame_ = None

        self.data_mutex_ = threading.Lock()

        self.display_frame_ = None
        self.display_mutex_ = threading.Lock()

        self.loop_closure_kfs_ = set()

    def mark_loop_closure(self, kf_id_a, kf_id_b):
        with self.data_mutex_:
            self.loop_closure_kfs_.add(kf_id_a)
            self.loop_closure_kfs_.add(kf_id_b)

    def get_loop_closure_kfs(self):
        with self.data_mutex_:
            return set(self.loop_closure_kfs_)

    def set_display_frame(self, frame):
        with self.display_mutex_:
            self.display_frame_ = frame

    def _get_display_snapshot(self):
        with self.display_mutex_:
            f = self.display_frame_
            if f is None:
                return None, [], None
            img = getattr(f, 'color_left_img_', None)
            if img is None:
                img = getattr(f, 'clean_left_img_', None)
            if img is None:
                img = getattr(f, 'left_img_', None)
            features = list(f.features_left_)
            frame_id = f.id_
        pts = [
            (int(feat.position_.pt[0]), int(feat.position_.pt[1]),
             feat.map_point_ is not None and not feat.is_outlier_)
            for feat in features
            if feat.position_ is not None
        ]
        return img, pts, frame_id

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
            html.Div([
                html.Button("Save view", id='save-view-btn', n_clicks=0),
                html.Span(id='save-view-status', style={'marginLeft': '10px'}),
            ], style={'marginBottom': '8px'}),
            dcc.Graph(id='slam-3d', style={'height': '85vh'}),
            dcc.Store(id='camera-store', data=_load_saved_camera()),
            dcc.Interval(id='interval', interval=2000, n_intervals=0),
        ])

        @self.dash_app.callback(
            Output('slam-3d', 'figure'),
            Input('interval', 'n_intervals')
        )
        def update_plot(_):
            return self._build_figure()

        @self.dash_app.callback(
            Output('camera-store', 'data'),
            Input('slam-3d', 'relayoutData'),
            State('camera-store', 'data'),
            prevent_initial_call=True,
        )
        def remember_camera(relayout, current):
            if relayout and 'scene.camera' in relayout:
                return relayout['scene.camera']
            return current

        @self.dash_app.callback(
            Output('save-view-status', 'children'),
            Input('save-view-btn', 'n_clicks'),
            State('camera-store', 'data'),
            prevent_initial_call=True,
        )
        def save_view(_n, camera):
            if not camera:
                return "No view to save yet — rotate the plot first."
            try:
                with open(VIEW_FILE, 'w') as f:
                    json.dump(camera, f)
                return f"View saved at {time.strftime('%H:%M:%S')}"
            except OSError as e:
                return f"Error: {e}"

        dash_thread = threading.Thread(
            target=self.dash_app.run,
            kwargs={'host': '0.0.0.0', 'port': 8050, 'debug': False},
            daemon=True
        )
        dash_thread.start()
        print("[Map] Dash server started at http://0.0.0.0:8050")

    # ------------------------------------------------------------------
    # Dash live 2-D frame viewer (current frame + landmark overlay)
    # ------------------------------------------------------------------

    def start_frame_server(self, port=8052):
        self.frame_app = Dash(__name__ + '_frames')
        self.frame_app.layout = html.Div([
            html.H2("SLAM - Current Frame"),
            html.Div(id='frame-header'),
            html.Img(id='frame-img', style={'width': '100%', 'imageRendering': 'pixelated'}),
            dcc.Interval(id='frame-interval', interval=500, n_intervals=0),
        ])

        @self.frame_app.callback(
            [Output('frame-img', 'src'),
             Output('frame-header', 'children')],
            Input('frame-interval', 'n_intervals'),
        )
        def update_frame(_):
            img, pts, frame_id = self._get_display_snapshot()
            if img is None:
                return '', 'Waiting for first frame...'

            if img.ndim == 2:
                vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            else:
                vis = img.copy()

            n_landmarks = 0
            for x, y, has_mp in pts:
                if has_mp:
                    cv.circle(vis, (x, y), 4, (0, 255, 0), -1)
                    n_landmarks += 1
                else:
                    cv.circle(vis, (x, y), 2, (0, 0, 255), 1)

            ok, buf = cv.imencode('.jpg', vis, [cv.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return '', 'JPEG encoding failed'
            b64 = base64.b64encode(buf.tobytes()).decode('ascii')
            header = f'Frame {frame_id} — {n_landmarks}/{len(pts)} landmarks'
            return f'data:image/jpeg;base64,{b64}', header

        frame_thread = threading.Thread(
            target=self.frame_app.run,
            kwargs={'host': '0.0.0.0', 'port': port, 'debug': False},
            daemon=True
        )
        frame_thread.start()
        print(f"[Map] Frame server started at http://0.0.0.0:{port}")

    def _build_figure(self):
        keyframes = self.get_all_keyframes()
        all_points = self.get_all_map_points()

        # --- Trajectory trace (always present, empty if no data) ---
        traj_x, traj_y, traj_z = [], [], []
        start_x, start_y, start_z = [], [], []
        marker_colors = []
        loop_kfs = self.get_loop_closure_kfs()
        n_loops = 0

        if keyframes:
            cam_pos = []
            for kf in keyframes:
                T_wc = np.linalg.inv(kf.pose_)
                cam_pos.append(T_wc[:3, 3])
            cam_pos = np.array(cam_pos)
            traj_x, traj_y, traj_z = cam_pos[:, 0], cam_pos[:, 2], cam_pos[:, 1]
            start_x, start_y, start_z = [cam_pos[0, 0]], [cam_pos[0, 2]], [cam_pos[0, 1]]
            marker_colors = ['red' if kf.id_ in loop_kfs else 'blue' for kf in keyframes]
            n_loops = sum(1 for kf in keyframes if kf.id_ in loop_kfs)

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
                marker=dict(size=4, color=marker_colors if marker_colors else 'blue'),
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

        scene = dict(
            xaxis=dict(title='X', range=[-5, 5]),
            yaxis=dict(title='Z (depth)', range=[-5, 5]),
            zaxis=dict(title='Y', range=[-5, 5]),
            aspectmode='cube',
            dragmode='orbit',
            uirevision='slam-scene',
        )
        saved_camera = _load_saved_camera()
        if saved_camera:
            scene['camera'] = saved_camera

        fig.update_layout(
            title=f'SLAM Live - {n_kf} KFs ({n_loops} loop), {n_pts} pts',
            scene=scene,
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
        
