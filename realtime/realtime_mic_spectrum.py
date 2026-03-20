import csv
import os
import queue
import wave
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from utils import save_wav

# =========================
# Configuration
# =========================
SAMPLE_RATE = 192000       # Sample rate in Hz
BLOCK_SIZE = 192000        # Number of samples per callback block
CHANNELS = 1              # Laptop mic is typically mono
DEVICE = 1                # None = system default input device
SAVE_AUDIO = True         # Save recorded audio to WAV when window closes
SAVE_SPECTRUM = True      # Save frame-by-frame spectrum to NPZ/CSV
OUTPUT_DIR = "output"     # Output directory for generated files

# Thread-safe queue for audio blocks from callback
audio_q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """Audio callback: push each incoming block to queue."""
    if status:
        print(status)
    # indata shape: (frames, channels)
    audio_q.put(indata[:, 0].copy())

def main():
    # Set up plot
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("Real-time Microphone Analysis")

    # Time-domain plot
    t = np.arange(BLOCK_SIZE) / SAMPLE_RATE
    time_line, = ax_time.plot(t, np.zeros(BLOCK_SIZE))
    ax_time.set_xlim(0, BLOCK_SIZE / SAMPLE_RATE)
    ax_time.set_ylim(-1.0, 1.0)
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_title("Waveform")

    # Frequency-domain plot
    freqs = np.fft.rfftfreq(BLOCK_SIZE, d=1.0 / SAMPLE_RATE)
    freq_line, = ax_freq.plot(freqs, np.zeros_like(freqs))
    ax_freq.set_xlim(0, SAMPLE_RATE / 2)
    ax_freq.set_ylim(-120, 0)  # dB range
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Magnitude (dB)")
    ax_freq.set_title("Spectrum (FFT)")

    window = np.hanning(BLOCK_SIZE)
    dominant_text = ax_freq.text(
        0.02, 0.92, "Dominant: -- Hz", transform=ax_freq.transAxes
    )

    # Buffers for saving
    recorded_blocks = []
    spectrum_frames = []
    dominant_freqs = []
    frame_times = []
    processed_blocks = 0

    def update(_frame):
        nonlocal processed_blocks
        # If data is available, take the latest block
        if not audio_q.empty():
            block = audio_q.get()
            processed_blocks += 1
            current_time = processed_blocks * (BLOCK_SIZE / SAMPLE_RATE)

            # Update time-domain data
            time_line.set_ydata(block)

            # FFT with Hann window, then convert magnitude to dB
            spec = np.fft.rfft(block * window)
            mag = np.abs(spec) + 1e-10
            mag_db = 20 * np.log10(mag)
            freq_line.set_ydata(mag_db)

            # Dominant frequency (ignore DC component at index 0)
            peak_idx = np.argmax(mag[1:]) + 1
            dom_freq = freqs[peak_idx]
            dominant_text.set_text(f"Dominant: {dom_freq:.1f} Hz")

            # Cache data for file outputs
            if SAVE_AUDIO:
                recorded_blocks.append(block.astype(np.float32))
            if SAVE_SPECTRUM:
                spectrum_frames.append(mag_db.astype(np.float32))
                dominant_freqs.append(float(dom_freq))
                frame_times.append(float(current_time))

        return time_line, freq_line, dominant_text

    # Start input stream
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
        device=DEVICE,
    ):
        _ani = FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)
        plt.tight_layout()
        plt.show()

    # Save files after the window is closed
    if SAVE_AUDIO or SAVE_SPECTRUM:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if SAVE_AUDIO and recorded_blocks:
            audio = np.concatenate(recorded_blocks)
            wav_path = os.path.join(OUTPUT_DIR, f"mic_audio_{stamp}.wav")
            save_wav(wav_path, audio, SAMPLE_RATE)
            print(f"Audio saved: {wav_path}")

        if SAVE_SPECTRUM and spectrum_frames:
            spectrum_array = np.vstack(spectrum_frames)
            times_array = np.array(frame_times, dtype=np.float32)
            dom_array = np.array(dominant_freqs, dtype=np.float32)

            npz_path = os.path.join(OUTPUT_DIR, f"spectrum_{stamp}.npz")
            np.savez(
                npz_path,
                sample_rate=SAMPLE_RATE,
                block_size=BLOCK_SIZE,
                freqs=freqs.astype(np.float32),
                times=times_array,
                dominant_freqs=dom_array,
                spectrum_db=spectrum_array,
            )
            print(f"Spectrum matrix saved: {npz_path}")

            csv_path = os.path.join(OUTPUT_DIR, f"dominant_freq_{stamp}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["time_s", "dominant_freq_hz"])
                writer.writerows(zip(times_array, dom_array))
            print(f"Dominant frequency track saved: {csv_path}")


if __name__ == "__main__":
    print("Starting real-time microphone spectrum analyzer.")
    print("Close the plot window to exit.")
    print("Default device setting:", sd.default.device)
    main()