import wave
import time
import numpy as np
import pandas as pd
from scipy import signal
from .timer import time_it


def save_wav(path, audio_float, sample_rate):
    """Save normalized float32 audio [-1,1] to mono 16-bit PCM WAV."""
    if audio_float.size == 0:
        return
    audio_clipped = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def save_angle(path, angles, begin_time, end_time):
    """Save measured angles and associated time windows to CSV."""
    data = {"begin_time": begin_time, "end_time": end_time, "angle": angles}
    pd.DataFrame(data).to_csv(path, index=False)

def correlation(first_sample, second_sample):
    """Calcule la corrélation entre deux signaux X et Y de même taille et renvoie le tau pour lequel on a le max de corrélation."""
    taille = len(first_sample)
    mean_first, standard_first = np.mean(first_sample), np.var(first_sample)
    mean_second, standard_second = np.mean(second_sample), np.var(second_sample)
    normalized_first,normalized_second = (first_sample - mean_first)/standard_first,(second_sample - mean_second)/standard_second
    correlation = signal.correlate(normalized_first,normalized_second,mode="full",method="fft")
    max_cor = np.argmax(correlation)
    delay_center = max_cor - taille
    return delay_center

def calculate_angle(delay_samples, sample_rate, distance_microphones, celerity):
    """Calculate the angle between hydrophone baseline normal and beacon."""
    delta_t = delay_samples / sample_rate
    ratio = (celerity * delta_t) / distance_microphones
    ratio = np.clip(ratio, -1.0, 1.0)
    angle = np.arcsin(ratio) * 180 / np.pi
    return angle if abs(angle)!=90.0 else None
