import numpy as np
import gtsam
from gtsam import symbol_shorthand, Pose3, Rot3, Point3

L = symbol_shorthand.L
X = symbol_shorthand.X


class Backend:

    def __init__(self, params_stereo, slam_map):
        self.map_ = slam_map

        # Calibration from the rectified projection matrix P1
        # (features are detected on rectified images, so P1 is the correct intrinsic)
        P1 = params_stereo['P1']
        fx, fy = P1[0, 0], P1[1, 1]
        s = P1[0, 1]
        cx, cy = P1[0, 2], P1[1, 2]
        self.calibration = gtsam.Cal3_S2(fx, fy, s, cx, cy)

        # --- Noise models ---
        # Prior on first pose (anchors the coordinate frame)
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])  # rot, trans
        )
        # Odometry between consecutive keyframes
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, 0.05, 0.01, 0.1])  # rot, trans
        )
        # Projection factor (pixel-space) with Huber robust kernel
        pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)
        huber = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
        self.projection_noise = gtsam.noiseModel.Robust.Create(huber, pixel_noise)
        # Very loose prior on landmarks (prevents indeterminate system for
        # landmarks observed from a single keyframe only)
        self.landmark_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 100.0)

        # --- iSAM2 ---
        isam_params = gtsam.ISAM2Params()
        isam_params.setRelinearizeThreshold(0.1)
        isam_params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(isam_params)

        # --- Bookkeeping ---
        self.processed_keyframe_ids = []
        self.initialized_landmarks = set()
        self.marginalized_landmarks = set()
        self.landmark_last_seen = {}  # landmark_id -> keyframe id when last observed
        self.max_landmarks = 10000
        self.result = None


    # ------------------------------------------------------------------
    # Pose convention helpers
    # Frontend stores T_cw (world-to-camera). GTSAM Pose3 = T_wc (camera-in-world).
    # ------------------------------------------------------------------

    def _to_gtsam_pose(self, T_cw):
        T_wc = np.linalg.inv(T_cw)
        return Pose3(Rot3(T_wc[:3, :3]), Point3(T_wc[:3, 3]))

    def _to_T_cw(self, gtsam_pose):
        T_wc = gtsam_pose.matrix()
        return np.linalg.inv(T_wc)

    # ------------------------------------------------------------------
    # Core optimisation
    # ------------------------------------------------------------------

    def update_map(self):
        """Interface compatible with Backend.py (g2o) — called by Frontend."""
        self.process_new_keyframes()

    def process_new_keyframes(self):
        """Incrementally add new keyframes to the factor graph and optimise."""
        all_kfs = self.map_.get_all_keyframes()
        new_kfs = [kf for kf in all_kfs if kf.id_ not in self.processed_keyframe_ids]
        if not new_kfs:
            return
        new_kfs.sort(key=lambda kf: kf.id_)

        new_factors = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()

        for kf in new_kfs:
            pose_key = X(kf.id_)
            gtsam_pose = self._to_gtsam_pose(kf.pose_)

            # Prior on the very first keyframe
            if len(self.processed_keyframe_ids) == 0:
                new_factors.add(
                    gtsam.PriorFactorPose3(pose_key, gtsam_pose, self.prior_noise))

            # Odometry to the previous keyframe
            if len(self.processed_keyframe_ids) > 0:
                prev_id = self.processed_keyframe_ids[-1]
                prev_kf = next(k for k in all_kfs if k.id_ == prev_id)
                prev_gtsam = self._to_gtsam_pose(prev_kf.pose_)
                relative = prev_gtsam.between(gtsam_pose)
                new_factors.add(gtsam.BetweenFactorPose3(
                    X(prev_id), pose_key, relative, self.odometry_noise))

            # Initial estimate for this pose
            new_values.insert(pose_key, gtsam_pose)

            # Projection factors for every observed landmark
            for feat in kf.features_left_:
                if feat.map_point_ is None or feat.is_outlier_:
                    continue

                mp = feat.map_point_

                # Skip landmarks that have been marginalized out of the graph
                if mp.id_ in self.marginalized_landmarks:
                    continue

                lm_key = L(mp.id_)
                u, v = feat.position_.pt
                measurement = np.array([u, v])

                new_factors.add(gtsam.GenericProjectionFactorCal3_S2(
                    measurement, self.projection_noise,
                    pose_key, lm_key, self.calibration,
                    False, False))  # throwCheirality, verboseCheirality

                self.landmark_last_seen[mp.id_] = kf.id_

                # First time seeing this landmark -> add initial value + loose prior
                if mp.id_ not in self.initialized_landmarks:
                    new_values.insert(lm_key, Point3(mp.pos_))
                    new_factors.add(gtsam.PriorFactorPoint3(
                        lm_key, Point3(mp.pos_), self.landmark_prior_noise))
                    self.initialized_landmarks.add(mp.id_)

            self.processed_keyframe_ids.append(kf.id_)

        # --- Run iSAM2 ---
        try:
            self.isam.update(new_factors, new_values)
            self.isam.update()  # extra Gauss-Newton iteration
            self.result = self.isam.calculateEstimate()
            self._update_map(all_kfs)
            print(f"[Backend] Optimised: {len(self.processed_keyframe_ids)} poses, "
                  f"{len(self.initialized_landmarks)} landmarks")
        except Exception as e:
            print(f"[Backend] Optimisation error: {e}")

    def _update_map(self, keyframes):
        """Write optimised poses and landmarks back into the map."""
        if self.result is None:
            return

        for kf in keyframes:
            key = X(kf.id_)
            if self.result.exists(key):
                kf.pose_ = self._to_T_cw(self.result.atPose3(key))

        for mp in self.map_.get_all_map_points():
            key = L(mp.id_)
            if self.result.exists(key):
                mp.pos_ = self.result.atPoint3(key)

