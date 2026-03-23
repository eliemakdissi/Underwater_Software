import wave

import numpy as np
import pandas as pd
from scipy import signal


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


def calculate_angle(first_sample, second_sample, sample_rate, distance_microphones, celerity):
    """Calculate the angle between hydrophone baseline normal and beacon."""
    correlation = signal.correlate(first_sample, second_sample, mode="full")
    max_corr = np.argmax(correlation)
    center = len(first_sample) - 1
    delay_samples = max_corr - center
    delta_t = delay_samples / sample_rate
    ratio = (celerity * delta_t) / distance_microphones
    ratio = np.clip(ratio, -1.0, 1.0)
    return (np.pi / 2 - np.arcsin(ratio)) * 180 / np.pi


def detect_interest_noise(first_sample, detection_threshold):
    """Detect if a block amplitude passes the detection threshold."""
    peak = np.max(np.abs(first_sample))
    return bool(peak > detection_threshold)


def bandpass_filter(data, target_freq, sample_rate, relative_margin=0.03, order=5):
    """Filter a given signal with a selective bandpass filter."""
    lowcut, highcut = target_freq * (1 - relative_margin), target_freq * (1 + relative_margin)
    sos = signal.butter(order, [lowcut, highcut], fs=sample_rate, btype="band", output="sos")
    return signal.sosfilt(sos, data)
