import glob
import cv2
import numpy as np
from vocabulary_tree import VocabularyTree


def extract_sift_descriptors(image_paths, max_images=None):
    sift = cv2.SIFT_create()
    all_desc = []

    if max_images is not None:
        image_paths = image_paths[:max_images]

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        kpts, desc = sift.detectAndCompute(img, None)
        if desc is None or len(desc) == 0:
            continue

        all_desc.append(desc.astype(np.float32))

    if len(all_desc) == 0:
        raise RuntimeError("Aucun descripteur SIFT trouvé pour entraîner le vocabulaire.")

    return np.vstack(all_desc)


if __name__ == "__main__":
    image_paths = glob.glob("/Users/pgpetitmangin/underwater/Underwater_Software/SLAM/images_test/image_caillou/*.jpg")

    descriptors = extract_sift_descriptors(image_paths, max_images=200)

    tree = VocabularyTree(k=10, max_depth=4)
    tree.fit(descriptors)
    tree.save("mon_vocabulaire_sift.pkl")

    print("Vocabulaire sauvegardé dans mon_vocabulaire_sift.pkl")