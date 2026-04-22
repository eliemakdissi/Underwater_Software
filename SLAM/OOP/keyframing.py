import cv2
import numpy as np


def should_create_keyframe(last_kf_descriptors, current_descriptors, threshold=0.3, min_matches=20):
    """
    Décide si la frame courante doit devenir une keyframe.
    Version SIFT: matching en L2.
    """
    if last_kf_descriptors is None or current_descriptors is None:
        return True

    if len(last_kf_descriptors) == 0 or len(current_descriptors) == 0:
        return True

    last_kf_descriptors = np.asarray(last_kf_descriptors, dtype=np.float32)
    current_descriptors = np.asarray(current_descriptors, dtype=np.float32)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(last_kf_descriptors, current_descriptors)

    if len(matches) < min_matches:
        return True

    matches = sorted(matches, key=lambda m: m.distance)

    # Seuil simple pour garder les meilleurs matches
    good_matches = [m for m in matches if m.distance < 250]

    match_ratio = len(good_matches) / max(len(last_kf_descriptors), 1)

    return match_ratio < threshold