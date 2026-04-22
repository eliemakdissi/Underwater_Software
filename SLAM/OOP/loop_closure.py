import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity


np.random.seed(42)
NUM_WORDS = 50 #taille du dictionnaire
inverted_index :dict = {i: [] for i in range(NUM_WORDS)}

def add_to_index(image_id, words_in_image):
    """Quand on traite une nouvelle image, on met à jour l'index"""
    unique_words = set(words_in_image)
    for word in unique_words:
        inverted_index[word].append(image_id)

def fast_search(new_image_words):
    """Trouver les images candidates sana parcourir les 10 000 images"""
    candidates = set()
    #on regarde les images qui ont un mot en commun au moins 
    for word in new_image_words:
        candidates.update(inverted_index[word])
    return candidates 


image_descriptors :list[int] = [] #data 






#empiler descriptors
all_descriptors = np.vstack(image_descriptors)


kmeans = KMeans(n_clusters=NUM_WORDS, random_state=42, n_init=10)
kmeans.fit(all_descriptors)


raw_histograms = [] 

for desc in image_descriptors:
    words = kmeans.predict(desc) #prediction pour chaque mots 
    
    hist, _ = np.histogram(words, bins=np.arange(NUM_WORDS + 1))
    raw_histograms.append(hist)

 # Matrice [nombre_images, NUM_WORDS]





# Le TF-IDF va donner moins d'importance aux mots qui apparaissent partout et plus d'importance aux mots rares et distinctifs.

tfidf_transformer = TfidfTransformer(norm='l1') 
tfidf_histograms = tfidf_transformer.fit_transform(raw_histograms).toarray()


#calcul des similarités

image_A = tfidf_histograms[0].reshape(1, -1)
image_B = tfidf_histograms[2].reshape(1, -1)


score = cosine_similarity(image_A, image_B)[0][0]

print(f"-> Score de similarité (Image 0 vs Image 2) : {score:.4f}")

# En pratique (comme expliqué dans la section 10.5.2 de ton texte), 
# on ne compare pas à un seuil fixe, mais au score de l'image précédente (t-1).
SEUIL_DE_BOUCLE = 0.8 
if score > SEUIL_DE_BOUCLE:
    print("Boucle")
else:
    print("Pas de boucle")


from sklearn.cluster import MiniBatchKMeans
import numpy as np

class VocabularyTree:
    def __init__(self, k=10, max_depth=5):
        self.k = k                  # branches par noeud
        self.max_depth = max_depth  
        self.model = None           # Le modèle KMeans pour ce niveau
        self.children = []          # sous arbres
        self.is_leaf = False        # Vrai si on est tout en bas de l'arbre
        self.word_id = None         # Identifiant du mot final (seulement pour les feuilles)

    def fit(self, descriptors, current_depth=1, word_counter=[0]):
        """Construit l'arbre récursivement."""
        # Condition d'arrêt : on a atteint la profondeur max ou il y a trop peu de données
        if current_depth == self.max_depth or len(descriptors) <= self.k:
            self.is_leaf = True
            self.word_id = word_counter[0]
            word_counter[0] += 1
            return

        # 1. On sépare les données actuelles en 'k' groupes
        # (MiniBatchKMeans est beaucoup plus rapide que KMeans classique sur de grosses données)
        self.model = MiniBatchKMeans(n_clusters=self.k, random_state=42, n_init=3)
        labels = self.model.fit_predict(descriptors)

        # 2. On crée 'k' sous-arbres (enfants) et on leur envoie leurs données respectives
        for i in range(self.k):
            child = VocabularyTree(k=self.k, max_depth=self.max_depth)
            child_descriptors = descriptors[labels == i]
            
            if len(child_descriptors) > 0:
                child.fit(child_descriptors, current_depth + 1, word_counter)
            
            self.children.append(child)

    def get_word(self, descriptor):
        """Descend l'entonnoir de l'arbre pour trouver le mot d'un descripteur en temps logarithmique."""
        if self.is_leaf:
            return self.word_id
            
        # On trouve la branche la plus proche à ce niveau
        branch_idx = self.model.predict([descriptor])[0]
        
        # S'il y a un enfant dans cette branche, on descend dedans
        if branch_idx < len(self.children):
             return self.children[branch_idx].get_word(descriptor)
        return None
    
import cv2
import numpy as np

def should_create_keyframe(last_kf_descriptors, current_descriptors, threshold=0.6):
    """
    Décide si la frame actuelle est assez différente de la dernière Keyframe.
    """
    if last_kf_descriptors is None or current_descriptors is None:
        return True # La toute première frame est TOUJOURS une Keyframe

    # On utilise le Matcher Bruteforce d'OpenCV pour comparer les points (ORB utilise NORM_HAMMING)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(last_kf_descriptors, current_descriptors)

    # Ratio de points qui correspondent encore
    match_ratio = len(matches) / max(len(last_kf_descriptors), 1)

    # Si on a perdu trop de correspondances (ex: ratio < 60%), 
    # c'est que le robot a beaucoup avancé ou tourné.
    if match_ratio < threshold:
        return True # Il est temps de créer une nouvelle Keyframe !
    
    return False # La vue n'a pas assez changé, on jette cette frame.

# --- Simulation d'un flux vidéo ---
print("--- Traitement Vidéo (30 FPS) ---")
last_kf_desc = None
keyframes_database = []

# Imaginons une boucle qui lit ta caméra frame par frame
for frame_idx in range(1, 101): # On simule 100 frames
    
    
    current_desc = np.random.randint(0, 256, (500, 32), dtype=np.uint8) 
    
    if should_create_keyframe(last_kf_desc, current_desc, threshold=0.15):
        print(f" Frame {frame_idx:03d} : Assez de mouvement -> Ajoutée comme KEYFRAME.")
        keyframes_database.append(frame_idx)
        last_kf_desc = current_desc
        
        