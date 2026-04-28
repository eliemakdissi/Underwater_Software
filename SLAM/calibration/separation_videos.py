import argparse
import os
import sys

import cv2


def split_stereo_video(input_path, output_left, output_right, fourcc_str="mp4v"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{input_path}'.")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width != 3840 or height != 1080:
        print(f"Warning: expected 3840x1080, got {width}x{height}. Proceeding by splitting width at the midpoint.")

    half_w = width // 2
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

    writer_l = cv2.VideoWriter(output_left,  fourcc, fps, (half_w, height))
    writer_r = cv2.VideoWriter(output_right, fourcc, fps, (half_w, height))

    if not writer_l.isOpened() or not writer_r.isOpened():
        print("Error: could not open output writers. Try a different codec (e.g. 'avc1', 'XVID').")
        cap.release()
        writer_l.release()
        writer_r.release()
        return False

    print(f"Splitting '{input_path}' ({width}x{height}@{fps:.2f}fps, {total_frames} frames)")
    print(f"  -> left:  {output_left}")
    print(f"  -> right: {output_right}")

    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer_l.write(frame[:, :half_w])
        writer_r.write(frame[:, half_w:])
        written += 1
        if total_frames > 0 and written % 100 == 0:
            print(f"  processed {written}/{total_frames}")

    cap.release()
    writer_l.release()
    writer_r.release()
    print(f"Done. Wrote {written} frames to each output.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split a side-by-side stereo video (default 3840x1080) into two single-camera videos."
    )
    parser.add_argument("input", help="Path to the input stereo video.")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Directory for output videos. Defaults to the input's directory."
    )
    parser.add_argument(
        "--codec", default="mp4v",
        help="FourCC codec for the writers (default: mp4v). Examples: mp4v, avc1, XVID, MJPG."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    stem, _ = os.path.splitext(os.path.basename(args.input))
    left_path  = os.path.join(out_dir, f"{stem}_left.mp4")
    right_path = os.path.join(out_dir, f"{stem}_right.mp4")

    ok = split_stereo_video(args.input, left_path, right_path, fourcc_str=args.codec)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
