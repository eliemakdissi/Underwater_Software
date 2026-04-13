import cv2
import os

def extract_and_split_frames(video_path, output_folder, num_frames=100):
    """
    Extrait un nombre donné d'images équidistantes d'une vidéo, les coupe en deux 
    (gauche/droite) et les sauvegarde dans un dossier.
    """
    # 1. Création du dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # 2. Ouverture de la vidéo
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video at '{video_path}'.")
        return

    # 3. Récupération des informations de la vidéo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print("Error: The video contains no frames or format is unsupported.")
        return

    # Vérification si le nombre demandé ne dépasse pas le nombre total de frames
    if num_frames > total_frames:
        print(f"Warning: Requested frames ({num_frames}) exceeds total frames ({total_frames}).")
        num_frames = total_frames

    # Calcul de l'intervalle (pas) pour obtenir des images équidistantes
    step = total_frames / num_frames

    print(f"Starting extraction of {num_frames} frames (left and right)...")

    # 4. Boucle d'extraction
    for i in range(num_frames):
        # Calcul de l'index exact de la frame pour rester équidistant
        frame_index = int(i * step)
        
        # Positionnement de la tête de lecture de la vidéo sur cette frame spécifique
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Lecture de l'image
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Cannot read frame at index {frame_index}.")
            continue

        # Optionnel : Vérification des dimensions (utile pour débugger)
        height, width, _ = frame.shape
        if width != 3840 or height != 1080:
            print(f"Warning: Dimensions of frame {i+1} are {width}x{height}, not 3840x1080.")

        # 5. Découpage de l'image en deux via les tableaux NumPy
        half_width = width // 2
        left_image = frame[:, :half_width]   # Garde toute la hauteur, prend la largeur de 0 à la moitié
        right_image = frame[:, half_width:]  # Garde toute la hauteur, prend la largeur de la moitié à la fin

        # 6. Sauvegarde des images
        # Formatage avec des zéros devant (ex: frame_0001_l.jpg) pour un tri correct dans les dossiers
        base_name = f"frame_{i+1:04d}" 
        
        left_path = os.path.join(output_folder, f"{base_name}_l.jpg")
        right_path = os.path.join(output_folder, f"{base_name}_r.jpg")

        cv2.imwrite(left_path, left_image)
        cv2.imwrite(right_path, right_image)

    # Libération des ressources de la vidéo
    cap.release()
    print(f"✅ Extraction complete! Images are saved in: {output_folder}")

# ==========================================
# UTILISATION
# ==========================================
if __name__ == "__main__":
    # Remplace par tes propres chemins
    MY_VIDEO_PATH = "SLAM/images_test/set_3_caillou/2026-03-27 14-48-19.mp4" 
    MY_OUTPUT_FOLDER = "SLAM/images_test/set_3_caillou"
    NUM_FRAMES_TO_EXTRACT = 100

    extract_and_split_frames(MY_VIDEO_PATH, MY_OUTPUT_FOLDER, num_frames=NUM_FRAMES_TO_EXTRACT)