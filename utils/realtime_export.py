from __future__ import annotations

import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

matplotlib.use("Agg")


def create_run_folder(output_root: str, prefix: str = "run") -> str:
    """Create and return one timestamped output folder for this session."""
    timestamp = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
    run_dir = os.path.join(output_root, f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def export_waveform_png(audio_mono: np.ndarray, sample_rate: float, output_dir: str) -> str:
    """Save full-session waveform plot."""
    path = os.path.join(output_dir, "waveform.png")
    t_axis = np.arange(len(audio_mono), dtype=np.float64) / float(sample_rate)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot(t_axis, audio_mono, color="#1f77b4", linewidth=0.6)
    ax.set_title("Waveform (full session)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def export_spectrogram_png(
    audio_mono: np.ndarray,
    sample_rate: float,
    fft_size: int,
    hop_size: int,
    output_dir: str,
) -> str | None:
    """Save full-session spectrogram plot."""
    path = os.path.join(output_dir, "spectrogram.png")
    nperseg = min(int(fft_size), len(audio_mono))
    if nperseg < 8:
        return None

    noverlap = min(nperseg - 1, max(0, nperseg - int(hop_size)))
    freqs, times, spec = spectrogram(
        audio_mono,
        fs=float(sample_rate),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        mode="magnitude",
    )
    spec_db = 20 * np.log10(spec + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    mesh = ax.pcolormesh(
        times,
        freqs / 1000.0,
        spec_db,
        shading="auto",
        cmap="inferno",
        vmin=-100,
        vmax=-20,
    )
    ax.set_title("Spectrogram (full session)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Power (dB)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
