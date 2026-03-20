import argparse
import glob
import os
import wave

import matplotlib.pyplot as plt
import numpy as np


def find_latest_file(pattern):
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def read_wav_mono(path):
    with wave.open(path, "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw = wav_file.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError("Only 16-bit PCM WAV is supported.")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels)[:, 0]
    return sample_rate, data


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Plot recorded microphone audio and spectrum outputs."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory containing mic_audio_*.wav, spectrum_*.npz, dominant_freq_*.csv",
    )
    parser.add_argument("--wav", default=None, help="Path to WAV file (optional)")
    parser.add_argument("--npz", default=None, help="Path to NPZ spectrum file (optional)")
    parser.add_argument("--csv", default=None, help="Path to dominant-freq CSV file (optional)")
    args = parser.parse_args()

    candidate_dirs = []
    if args.output_dir:
        candidate_dirs.append(os.path.abspath(args.output_dir))
    # Default location: same folder as this script.
    candidate_dirs.append(os.path.join(script_dir, "output"))
    # Backward compatibility: old location under Underwater_Software/output.
    candidate_dirs.append(os.path.normpath(os.path.join(script_dir, "..", "output")))

    # Remove duplicates while preserving order.
    candidate_dirs = list(dict.fromkeys(candidate_dirs))

    wav_path = args.wav
    npz_path = args.npz
    csv_path = args.csv
    used_out_dir = None

    # Auto-detect an output directory containing at least one WAV and one NPZ.
    if wav_path is None or npz_path is None or csv_path is None:
        for out_dir in candidate_dirs:
            detected_wav = wav_path or find_latest_file(os.path.join(out_dir, "mic_audio_*.wav"))
            detected_npz = npz_path or find_latest_file(os.path.join(out_dir, "spectrum_*.npz"))
            detected_csv = csv_path or find_latest_file(
                os.path.join(out_dir, "dominant_freq_*.csv")
            )
            if detected_wav is not None and detected_npz is not None:
                wav_path = detected_wav
                npz_path = detected_npz
                csv_path = detected_csv
                used_out_dir = out_dir
                break

    if wav_path is None or npz_path is None:
        raise FileNotFoundError(
            "Cannot find required files. Need at least one WAV and one NPZ output. "
            f"Searched directories: {candidate_dirs}"
        )

    sample_rate, audio = read_wav_mono(wav_path)
    wav_time = np.arange(audio.size) / float(sample_rate)

    npz_data = np.load(npz_path)
    freqs = npz_data["freqs"]
    times = npz_data["times"]
    dominant_freqs = npz_data["dominant_freqs"]
    spectrum_db = npz_data["spectrum_db"]

    # Prefer CSV if provided and valid; otherwise use NPZ dominant frequencies.
    csv_times = None
    csv_dom = None
    if csv_path and os.path.exists(csv_path):
        try:
            loaded = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            if loaded.ndim == 1 and loaded.size == 2:
                loaded = loaded.reshape(1, 2)
            if loaded.size > 0:
                csv_times = loaded[:, 0]
                csv_dom = loaded[:, 1]
        except Exception:
            csv_times = None
            csv_dom = None

    dom_t = csv_times if csv_times is not None else times
    dom_f = csv_dom if csv_dom is not None else dominant_freqs

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle("Recorded Audio and Spectrum")

    # 1) Waveform
    axes[0].plot(wav_time, audio, linewidth=0.8)
    axes[0].set_title("Waveform (WAV)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, wav_time[-1] if wav_time.size else 1)
    axes[0].set_ylim(-1.0, 1.0)

    # 2) Spectrogram from saved frame spectra
    # spectrum_db shape: [n_frames, n_bins], transpose for imshow -> [n_bins, n_frames]
    extent = [times[0] if times.size else 0, times[-1] if times.size else 1, freqs[0], freqs[-1]]
    im = axes[1].imshow(
        spectrum_db.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    axes[1].set_title("Spectrogram (from NPZ)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axes[1], label="Magnitude (dB)")

    # 3) Dominant frequency track
    axes[2].plot(dom_t, dom_f, linewidth=1.0)
    axes[2].set_title("Dominant Frequency Track")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    if dom_t is not None and np.size(dom_t) > 0:
        axes[2].set_xlim(float(np.min(dom_t)), float(np.max(dom_t)))
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Plotted files:")
    if used_out_dir:
        print(f"  Output dir: {used_out_dir}")
    print(f"  WAV: {wav_path}")
    print(f"  NPZ: {npz_path}")
    if csv_path:
        print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
