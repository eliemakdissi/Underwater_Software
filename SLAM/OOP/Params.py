import threading


DEFAULTS = {
    # Feature detection
    'sift_contrast_threshold': 0.01,
    'clahe_clip_limit': 2.0,

    # Matching / tracking
    'lowe_ratio': 0.80,
    'essential_ransac_threshold': 3.0,
    'pnp_reprojection_error': 5.0,
    'depth_min': 0.1,
    'depth_max': 15.0,

    # Statistical outlier removal (stereo triangulation)
    'sor_nb_neighbors': 15,
    'sor_std_ratio': 15.0,

    # Keyframe triggers
    'kf_min_inliers': 50,
    'kf_max_frames': 15,
    'kf_translation': 0.15,
    'kf_angle_deg': 8.0,

    # Map
    'num_active_keyframes': 7,

    # Backends
    'chi2_threshold': 5.991,
    'max_landmarks': 10000,
}


_lock = threading.Lock()
_params = dict(DEFAULTS)
_versions = {k: 0 for k in DEFAULTS}


def get(key):
    with _lock:
        return _params[key]


def set(key, value):
    with _lock:
        if key not in _params:
            raise KeyError(f"Unknown param '{key}'")
        current = _params[key]
        target_type = type(DEFAULTS[key])
        try:
            value = target_type(value)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot cast {value!r} to {target_type.__name__} for '{key}'")
        if value != current:
            _params[key] = value
            _versions[key] += 1


def version(key):
    with _lock:
        return _versions[key]


def all():
    with _lock:
        return dict(_params)
