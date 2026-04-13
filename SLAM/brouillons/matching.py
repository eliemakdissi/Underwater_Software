#!/opt/homebrew/bin/python3.14

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Matching entre deux images avec ORB et BFMatcher."
    )
    parser.add_argument(
        "--input1",
        default=str(script_dir / "image1.jpg"),
        help="Chemin vers la premiere image.",
    )
    parser.add_argument(
        "--input2",
        default=str(script_dir / "image2.jpg"),
        help="Chemin vers la deuxieme image.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="N'ouvre pas de fenetre d'affichage.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Chemin optionnel du rendu des matches.",
    )
    parser.add_argument("--clip-limit", type=float, default=5.0)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--ratio-thresh", type=float, default=0.7)
    parser.add_argument("--nfeatures", type=int, default=1500)
    parser.add_argument("--scale-factor", type=float, default=1.2)
    parser.add_argument("--nlevels", type=int, default=8)
    parser.add_argument("--edge-threshold", type=int, default=10)
    parser.add_argument("--wta-k", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--fast-threshold", type=int, default=20)
    return parser


def load_grayscale_image(path_str: str) -> cv2.typing.MatLike | None:
    return cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)


def main() -> int:
    args = build_parser().parse_args()

    img1 = load_grayscale_image(args.input1)
    img2 = load_grayscale_image(args.input2)
    if img1 is None or img2 is None:
        print("Impossible de lire les images :")
        print(f"  input1 = {args.input1}")
        print(f"  input2 = {args.input2}")
        return 1

    clahe = cv2.createCLAHE(
        clipLimit=args.clip_limit,
        tileGridSize=(args.tile_size, args.tile_size),
    )
    img1_enhanced = clahe.apply(img1)
    img2_enhanced = clahe.apply(img2)

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

    keypoints1, descriptors1 = orb.detectAndCompute(img1_enhanced, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_enhanced, None)

    if descriptors1 is None or descriptors2 is None:
        print("Descripteurs vides, matching impossible.")
        return 1

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for match_candidates in knn_matches:
        if len(match_candidates) == 2:
            best_match, second_match = match_candidates
            if best_match.distance < args.ratio_thresh * second_match.distance:
                good_matches.append(best_match)

    match_view = cv2.drawMatches(
        img1,
        keypoints1,
        img2,
        keypoints2,
        good_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(-1, -1, -1),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.putText(
        match_view,
        f"Good matches: {len(good_matches)} / {len(knn_matches)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    print(f"Keypoints image 1 : {len(keypoints1)}")
    print(f"Keypoints image 2 : {len(keypoints2)}")
    print(f"Good matches     : {len(good_matches)}")

    if args.output:
        output_path = Path(args.output)
        if cv2.imwrite(str(output_path), match_view):
            print(f"Rendu enregistre : {output_path}")
        else:
            print(f"Echec de l'enregistrement : {output_path}")
            return 1

    if not args.no_display:
        cv2.imshow("Good Matches", match_view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
