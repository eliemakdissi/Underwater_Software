import pickle
import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans


class VocabularyTree:
    def __init__(self, k=10, max_depth=5):
        self.k = k
        self.max_depth = max_depth
        self.model = None
        self.children = []
        self.is_leaf = False
        self.word_id = None

    def fit(self, descriptors, current_depth=1, word_counter=None):
        """
        Entraîne récursivement l'arbre sur des descripteurs float32 de shape [N, D].
        """
        if word_counter is None:
            word_counter = [0]

        descriptors = np.asarray(descriptors, dtype=np.float32)

        if len(descriptors) == 0:
            self.is_leaf = True
            self.word_id = word_counter[0]
            word_counter[0] += 1
            return

        if current_depth >= self.max_depth or len(descriptors) <= self.k:
            self.is_leaf = True
            self.word_id = word_counter[0]
            word_counter[0] += 1
            return

        n_clusters = min(self.k, len(descriptors))
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=3,
            batch_size=1024
        )

        labels = self.model.fit_predict(descriptors)
        self.children = []

        for i in range(n_clusters):
            child = VocabularyTree(k=self.k, max_depth=self.max_depth)
            child_descriptors = descriptors[labels == i]
            child.fit(child_descriptors, current_depth + 1, word_counter)
            self.children.append(child)

    def get_word(self, descriptor):
        """
        Retourne l'identifiant du mot visuel pour 1 descripteur.
        """
        descriptor = np.asarray(descriptor, dtype=np.float32)

        if self.is_leaf:
            return self.word_id

        if self.model is None or len(self.children) == 0:
            return None

        branch_idx = int(self.model.predict(descriptor.reshape(1, -1))[0])

        if 0 <= branch_idx < len(self.children):
            return self.children[branch_idx].get_word(descriptor)

        return None

    def transform(self, descriptors):
        """
        Convertit une liste de descripteurs en liste de mots visuels.
        """
        words = []
        descriptors = np.asarray(descriptors, dtype=np.float32)

        for d in descriptors:
            word = self.get_word(d)
            if word is not None:
                words.append(word)
        return words

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)