import csv
import json
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .legacy import save_wav

TARGET_FREQS_HZ = (8800.0, 37500.0)


def read_wav_mono(path):
    """Read 16-bit PCM WAV and return sample rate, mono float audio, channel count."""
    with wave.open(str(path), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw = wav_file.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported.")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    mono = data.reshape(-1, n_channels).mean(axis=1) if n_channels > 1 else data
    return sample_rate, mono, n_channels


def selective_bandpass(data, sample_rate, target_freq, relative_margin=0.03, order=6):
    lowcut = target_freq * (1.0 - relative_margin)
    highcut = target_freq * (1.0 + relative_margin)
    nyquist = sample_rate / 2.0
    if highcut >= nyquist:
        raise ValueError(
            f"Target {target_freq} Hz with margin {relative_margin} exceeds Nyquist ({nyquist} Hz)."
        )

    sos = signal.butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    return signal.sosfiltfilt(sos, data).astype(np.float32)


def apply_multi_band_filter(
    audio,
    sample_rate,
    target_freqs,
    relative_margin=0.03,
    order=6,
    band_gain=20.0,
):
    if not target_freqs:
        raise ValueError("target_freqs must contain at least one frequency.")
    if band_gain <= 0:
        raise ValueError("band_gain must be positive.")
    filtered = np.zeros_like(audio, dtype=np.float32)
    for target in target_freqs:
        filtered += selective_bandpass(audio, sample_rate, target, relative_margin, order)
    filtered *= float(band_gain)
    peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0
    return filtered / peak if peak > 1.0 else filtered


def process_audio(audio, sample_rate, block_size):
    if audio.size < block_size:
        audio = np.concatenate([audio, np.zeros(block_size - audio.size, dtype=np.float32)])

    n_blocks = int(np.ceil(audio.size / block_size))
    padded_size = n_blocks * block_size
    if padded_size > audio.size:
        audio = np.pad(audio, (0, padded_size - audio.size))

    blocks = audio.reshape(n_blocks, block_size)
    window = np.hanning(block_size).astype(np.float32)
    freqs = np.fft.rfftfreq(block_size, d=1.0 / sample_rate).astype(np.float32)

    spectrum_frames, dominant_freqs, frame_times = [], [], []
    for idx, block in enumerate(blocks):
        spec = np.fft.rfft(block * window)
        mag = np.abs(spec) + 1e-10
        spectrum_frames.append((20.0 * np.log10(mag)).astype(np.float32))
        peak_idx = int(np.argmax(mag[1:]) + 1) if mag.size > 1 else 0
        dominant_freqs.append(float(freqs[peak_idx]) if peak_idx < freqs.size else 0.0)
        frame_times.append(float((idx + 1) * (block_size / sample_rate)))

    return {
        "freqs": freqs,
        "times": np.array(frame_times, dtype=np.float32),
        "dominant_freqs": np.array(dominant_freqs, dtype=np.float32),
        "spectrum_db": np.vstack(spectrum_frames).astype(np.float32),
        "block_size": int(block_size),
    }


def save_analysis_outputs(
    input_wav,
    output_dir,
    sample_rate,
    channels,
    filtered_audio,
    processed,
    filtered_frequencies_hz,
    relative_margin,
    filter_order,
    band_gain,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_wav(str(output_dir / "filtered_signal.wav"), filtered_audio, sample_rate)

    np.savez(
        output_dir / "spectrum.npz",
        sample_rate=sample_rate,
        block_size=processed["block_size"],
        freqs=processed["freqs"],
        times=processed["times"],
        dominant_freqs=processed["dominant_freqs"],
        spectrum_db=processed["spectrum_db"],
    )

    with open(output_dir / "dominant_freq.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time_s", "dominant_freq_hz"])
        writer.writerows(zip(processed["times"], processed["dominant_freqs"]))

    wav_time = np.arange(filtered_audio.size) / float(sample_rate)
    times = processed["times"]
    freqs = processed["freqs"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle("Filtered Audio Analysis Results")

    axes[0].plot(wav_time, filtered_audio, linewidth=0.8)
    axes[0].set_title(f"Waveform (Filtered, full duration, gain x{band_gain:g})")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_ylim(-1.0, 1.0)
    if wav_time.size:
        axes[0].set_xlim(0, wav_time[-1])
    axes[0].text(
        0.02,
        0.92,
        f"Band gain: x{band_gain:g}",
        transform=axes[0].transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )

    extent = [
        float(times[0]) if times.size else 0.0,
        float(times[-1]) if times.size else 1.0,
        float(freqs[0]) if freqs.size else 0.0,
        float(freqs[-1]) if freqs.size else sample_rate / 2.0,
    ]
    image = axes[1].imshow(
        processed["spectrum_db"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    axes[1].set_title("Spectrogram")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(image, ax=axes[1], label="Magnitude (dB)")

    axes[2].plot(times, processed["dominant_freqs"], linewidth=1.0)
    axes[2].set_title("Dominant Frequency Track")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].grid(True, alpha=0.3)
    if times.size:
        axes[2].set_xlim(float(times[0]), float(times[-1]))

    fig.tight_layout()
    fig.savefig(output_dir / "analysis_plots.png", dpi=150)
    plt.close(fig)

    metadata = {
        "input_wav": str(input_wav),
        "sample_rate_hz": int(sample_rate),
        "input_channels": int(channels),
        "duration_s": float(filtered_audio.size / sample_rate) if sample_rate else 0.0,
        "filtered_frequencies_hz": [float(freq) for freq in filtered_frequencies_hz],
        "filter_relative_margin": float(relative_margin),
        "filter_order": int(filter_order),
        "band_gain": float(band_gain),
        "block_size_samples": int(processed["block_size"]),
        "num_frames": int(processed["times"].size),
        "generated_files": [
            "filtered_signal.wav",
            "spectrum.npz",
            "dominant_freq.csv",
            "analysis_plots.png",
        ],
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=2)


def find_input_wav(input_dir):
    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV file found in input folder: {input_dir}")
    return wav_files[0]
