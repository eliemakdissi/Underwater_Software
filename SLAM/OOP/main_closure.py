import numpy as np

from ops.cloud_sift import generate_cloud
from vocabulary_tree import VocabularyTree
from keyframing import should_create_keyframe
from database import LoopClosureDatabase
import glob
import cv2

def verify_loop_geometrically(current_kf, candidate_kf, min_common_words=15):
    current_words = set(current_kf["words"])
    candidate_words = set(candidate_kf["words"])
    common = len(current_words.intersection(candidate_words))
    return common >= min_common_words
def show_loop_match(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None or img2 is None:
        print("Erreur chargement images")
        return

    # Resize pour avoir même taille
    h = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))

    combined = cv2.hconcat([img1, img2])

    cv2.imshow("Loop Closure", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    vocab_tree = VocabularyTree.load("/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/mon_vocabulaire_sift.pkl")

    def collect_leaf_ids(node, out):
        if node.is_leaf:
            out.append(node.word_id)
        else:
            for child in node.children:
                collect_leaf_ids(child, out)

    leaf_ids = []
    collect_leaf_ids(vocab_tree, leaf_ids)
    num_words = max(leaf_ids) + 1 if len(leaf_ids) > 0 else 1

    db = LoopClosureDatabase(num_words=num_words)

    last_kf_desc = None

    left_images = sorted(glob.glob("/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/data_sortie_mer/frames/gauche/sortie_left.mp4_fixed/frames/*.jpg"))

    right_images = sorted(glob.glob("/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/data_sortie_mer/frames/droite/sortie_right.mp4_fixed/frames/*.jpg"))

    # On s'assure qu'on a le même nombre
    n = min(len(left_images), len(right_images))

    image_pairs = list(zip(left_images[:n], right_images[:n]))
    image_pairs = image_pairs[:3000:10]

    print(f"{len(image_pairs)} paires d'images chargées")

    for frame_id, (path_L, path_R) in enumerate(image_pairs):
        print(f"\n--- Traitement frame {frame_id} ---")

        try:
            p3d, pts2d, descriptors = generate_cloud(path_L, path_R)
        except Exception as e:
            print(f"Erreur generate_cloud: {e}")
            continue

        if descriptors is None or len(descriptors) == 0:
            print("Pas de descripteurs.")
            continue

        descriptors = descriptors.astype(np.float32)

        if not should_create_keyframe(last_kf_desc, descriptors, threshold=0.3):
            print("Frame ignorée (redondante).")
            continue

        print(f"Nouvelle keyframe : KF_{frame_id}")

        words_in_image = vocab_tree.transform(descriptors)

        if len(words_in_image) == 0:
            print("Aucun mot visuel trouvé.")
            continue

        candidates = db.find_loop_candidates(
            words_in_image,
            current_kf_id=frame_id,
            min_temporal_distance=2,
            top_k=3
        )

        kf = {
            "id": frame_id,
            "p3d": p3d,
            "pts2d": pts2d,
            "desc": descriptors,
            "words": words_in_image,
            "path_L": path_L,
        }

        if len(candidates) > 0:
            print("Candidats de boucle :", candidates)
            best_id, best_score = candidates[0]
            print(f"Meilleur candidat: KF_{best_id}, score={best_score:.4f}")

            best_id, best_score = candidates[0]
            candidate_kf = next(x for x in db.keyframes if x["id"] == best_id)
            if best_score > 0.8 and verify_loop_geometrically(kf, candidate_kf):
                print(f"BOUCLE DETECTEE avec KF_{best_id} | score={best_score:.4f}")
                print(f"Frame courante = KF_{frame_id} | boucle proposée avec KF_{best_id}")
                show_loop_match(kf["path_L"], candidate_kf["path_L"])
            else:
                print("Candidat rejeté après vérification.")
        else:
            print("Aucun candidat de boucle.")

        db.add_keyframe(kf)
        last_kf_desc = descriptors

    print(f"\nNombre total de keyframes stockées : {len(db.keyframes)}")


if __name__ == "__main__":
    main()