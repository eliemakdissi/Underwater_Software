# Implementation of the Map class
import threading
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

class Map:
    
    def __init__(self):

        self.landmarks_ = {}
        self.active_landmarks_ = {}
        self.keyframes_ = {}
        self.active_keyframes_ = {}

        self.num_active_keyframes_ = 7
        self.current_frame_ = None

        self.data_mutex_ = threading.Lock()

    def insert_keyframes(self, frame):

        with self.data_mutex_:
            self.current_frame_ = frame
            self.keyframes_[frame.id_] = frame
            self.active_keyframes_[frame.id_] = frame

        if len(self.active_keyframes_) > self.num_active_keyframes_:
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

        if len(self.active_keyframes_) <= self.num_active_keyframes_:
            return
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
            traj_x, traj_y, traj_z = cam_pos[:, 0], cam_pos[:, 2], -cam_pos[:, 1]
            start_x, start_y, start_z = [cam_pos[0, 0]], [cam_pos[0, 2]], [-cam_pos[0, 1]]

        # --- Points trace (always present, empty if no data) ---
        pts_x, pts_y, pts_z = [], [], []
        n_pts_filtered = 0

        if all_points:
            pts = np.array([mp.pos_ for mp in all_points if not mp.is_outlier_])
            if len(pts) > 0:
                mask = (pts[:, 2] > 0.05) & (pts[:, 2] < 20.0)
                pts = pts[mask]
                if len(pts) > 0:
                    pts_x, pts_y, pts_z = pts[:, 0], pts[:, 2], -pts[:, 1]
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
                marker=dict(size=1, color='red', opacity=0.3),
                name=f'Points ({n_pts_filtered})'
            ),
        ])

        fig.update_layout(
            title=f'SLAM Live - {n_kf} KFs, {n_pts} pts',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Z (depth)',
                zaxis_title='Y',
                aspectmode='data',
                uirevision='slam-scene',
            ),
            uirevision='slam',
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig
