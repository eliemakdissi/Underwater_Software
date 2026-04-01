import time
from functools import wraps

def time_it(func):
    """Décorateur pour mesurer le temps d'exécution d'une fonction en millisecondes."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        elapsed_ms = (end - start) * 1000
        # On affiche le nom de la fonction et son temps en ms
        print(f"⏱️ [{func.__name__}] : {elapsed_ms:.2f} ms")
        
        return result
    return wrapper