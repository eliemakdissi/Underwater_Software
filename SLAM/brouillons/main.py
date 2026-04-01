#!/opt/homebrew/bin/python3.14

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lecture camera/video avec ORB temps reel."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="Index de camera (ex: 0) ou chemin vers une video.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Traite les frames sans ouvrir de fenetre.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Nombre maximal de frames a traiter (0 = illimite).",
    )
    parser.add_argument("--clip-limit", type=float, default=3.5)
    parser.add_argument("--tile-size", type=int, default=6)
    parser.add_argument("--min-response-ratio", type=float, default=0.0)
    parser.add_argument("--nfeatures", type=int, default=1500)
    parser.add_argument("--scale-factor", type=float, default=1.2)
    parser.add_argument("--nlevels", type=int, default=8)
    parser.add_argument("--edge-threshold", type=int, default=10)
    parser.add_argument("--wta-k", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--fast-threshold", type=int, default=20)
    return parser


def open_capture(source_arg: str) -> tuple[cv2.VideoCapture, bool, str]:
    try:
        camera_index = int(source_arg)
        capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not capture.isOpened():
            capture = cv2.VideoCapture(camera_index)
        return capture, False, str(camera_index)
    except ValueError:
        capture = cv2.VideoCapture(source_arg)
        return capture, True, source_arg


def main() -> int:
    args = build_parser().parse_args()

    default_video_path = Path(__file__).with_name("video.mkv")
    source_arg = str(default_video_path) if args.source == "video.mkv" else args.source

    capture, use_video_file, source_label = open_capture(source_arg)
    if not capture.isOpened():
        print(f"Impossible d'ouvrir la source : {source_label}")
        return 1

    if not use_video_file:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_FPS, 30)

    wait_time_ms = 1
    fps = capture.get(cv2.CAP_PROP_FPS)
    if use_video_file and fps > 0:
        wait_time_ms = max(1, int(1000.0 / fps))

    clahe = cv2.createCLAHE(
        clipLimit=args.clip_limit,
        tileGridSize=(args.tile_size, args.tile_size),
    )
    orb = cv2.ORB.create(
        nfeatures=args.nfeatures,
        scaleFactor=args.scale_factor,
        nlevels=args.nlevels,
        edgeThreshold=args.edge_threshold,
        firstLevel=0,
        WTA_K=args.wta_k,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=args.patch_size,
        fastThreshold=args.fast_threshold,
    )

    window_name = "Flux video" if use_video_file else "Flux camera"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    processed_frames = 0
    print(f"Source ouverte : {source_label}")
    print("Appuie sur q ou Echap pour quitter.")

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            print("Fin de la video." if use_video_file else "Lecture frame impossible.")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_enhanced = clahe.apply(frame_gray)
        keypoints, descriptors = orb.detectAndCompute(frame_enhanced, None)

        detected_keypoints_count = len(keypoints)
        if keypoints:
            max_response = max(keypoint.response for keypoint in keypoints)
            min_accepted_response = max_response * args.min_response_ratio

            filtered_keypoints = []
            filtered_descriptor_rows = []
            for index, keypoint in enumerate(keypoints):
                if keypoint.response >= min_accepted_response:
                    filtered_keypoints.append(keypoint)
                    if descriptors is not None:
                        filtered_descriptor_rows.append(descriptors[index])

            keypoints = filtered_keypoints
            if filtered_descriptor_rows:
                descriptors = cv2.vconcat(filtered_descriptor_rows)
            else:
                descriptors = None

        frame_with_features = cv2.drawKeypoints(
            frame_enhanced,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
        )

        cv2.putText(
            frame_with_features,
            f"ORB gardes: {len(keypoints)} / {detected_keypoints_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        if not args.no_display:
            cv2.imshow(window_name, frame_with_features)
            key = cv2.waitKey(wait_time_ms) & 0xFF
            if key in (ord("q"), 27):
                break
        else:
            if args.max_frames > 0 and processed_frames + 1 >= args.max_frames:
                break

        processed_frames += 1
        if args.max_frames > 0 and processed_frames >= args.max_frames:
            break

    capture.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
