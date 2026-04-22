import numpy as np
from collections import defaultdict, Counter



    

   
class LoopClosureDatabase:
    def __init__(self, num_words):
        self.num_words = num_words
        self.keyframes = []
        self.inverted_index = defaultdict(set)   # word_id -> set(kf_id)
        self.word_doc_freq = Counter()           # nombre d'images contenant chaque mot
        self.histograms = {}                     # kf_id -> histogramme brut
        self.tfidf_histograms = {}               # kf_id -> histogramme tf-idf
    
    def get_keyframe(self, kf_id):
        """Récupère un keyframe par son ID."""
        for kf in self.keyframes:
            if kf["id"] == kf_id:
                return kf
        return None

    def _compute_histogram(self, words):
        hist = np.zeros(self.num_words, dtype=np.float32)
        for w in words:
            if 0 <= w < self.num_words:
                hist[w] += 1.0
        return hist

    def _compute_idf(self):
        """
        idf(w) = log((N + 1)/(df + 1)) + 1
        """
        N = max(len(self.keyframes), 1)
        idf = np.ones(self.num_words, dtype=np.float32)

        for w in range(self.num_words):
            df = self.word_doc_freq.get(w, 0)
            idf[w] = np.log((N + 1) / (df + 1)) + 1.0

        return idf

    def _tfidf_normalize(self, hist):
        if hist.sum() == 0:
            return hist.copy()

        tf = hist / hist.sum()
        idf = self._compute_idf()
        tfidf = tf * idf

        norm = np.linalg.norm(tfidf)
        if norm > 1e-8:
            tfidf = tfidf / norm
        return tfidf.astype(np.float32)

    def add_keyframe(self, kf):
        """
        kf = {
            'id': int,
            'p3d': ...,
            'pts2d': ...,
            'desc_binary': ...,
            'desc_float': ...,
            'words': list[int]
        }
        """
        kf_id = kf["id"]
        words = kf["words"]

        self.keyframes.append(kf)

        hist = self._compute_histogram(words)
        self.histograms[kf_id] = hist

        unique_words = set(words)
        for w in unique_words:
            self.inverted_index[w].add(kf_id)
            self.word_doc_freq[w] += 1

        # Recalcule les TF-IDF après ajout
        self._refresh_tfidf()

    def _refresh_tfidf(self):
        for kf in self.keyframes:
            kf_id = kf["id"]
            self.tfidf_histograms[kf_id] = self._tfidf_normalize(self.histograms[kf_id])

    def find_loop_candidates(self, words, current_kf_id, min_temporal_distance=10, top_k=5):
        """
        Retourne les meilleurs candidats de boucle par score cosinus.
        On exclut les keyframes trop proches temporellement.
        """
        query_hist = self._compute_histogram(words)
        query_tfidf = self._tfidf_normalize(query_hist)

        candidates = set()
        for w in set(words):
            candidates.update(self.inverted_index[w])

        filtered_candidates = [
            kf_id for kf_id in candidates
            if abs(kf_id - current_kf_id) >= min_temporal_distance
        ]

        if len(filtered_candidates) == 0:
            return []

        scored = []
        for kf_id in filtered_candidates:
            db_vec = self.tfidf_histograms[kf_id]
            score = float(np.dot(query_tfidf, db_vec))
            scored.append((kf_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]