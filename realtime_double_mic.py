import os
import queue
import wave
from datetime import datetime
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from utils import save_angle,save_wav,calculate_angle,detect_interest_noise

# =========================
# Configuration
# =========================
SAMPLE_RATE = 96000                     # Sample rate in Hz
BLOCK_SIZE = 0.5                        # Time per block in second
BLOCK_TIME = BLOCK_SIZE*SAMPLE_RATE     # Number of samples per callback block
DISPLAYED_TIME = 3                      # Duration in which the signal is displayed
NUM_SAMPLES_DISPLAYED = DISPLAYED_TIME*SAMPLE_RATE   # Number of samples displayed according to the time and sample rate
CHANNELS = 2                            # Bi-directionnal mic with Focusrite card
DEVICE =  1                             # None = system default input device, else check device with "python -m sounddevice"
SAVE_AUDIO = True                       # Save recorded audio to WAV when window closes
SAVE_ANGLES = True                      # Save measured angle to a csv file
OUTPUT_DIR = "output"                   # Output directory for generated files
DETECTION_THRESHOLD = 0.6               # Detection threshold for calculating angles
C_WATER = 1500                          # Celerity of sound in the medium tested in m/s
DISTANCE_MICROPHONES=1                  # Distance between microphones in m

# Thread-safe queue for audio blocks from callback
audio_q = queue.Queue()


def audio_callback(indata, status):
    """Audio callback: push each incoming block to queue."""
    if status:
        print(status)
    # indata shape: (frames, channels)
    audio_q.put(indata.copy())

def main():
    # Set up plot
    fig, (ax_first, ax_second) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("Real-time Microphone display")
    first_signal_plot,second_signal_plot = np.zeros(NUM_SAMPLES_DISPLAYED),np.zeros(NUM_SAMPLES_DISPLAYED)
    # Time plot - first input
    t = np.linspace(0,DISPLAYED_TIME,NUM_SAMPLES_DISPLAYED)
    time_first, = ax_first.plot(t, first_signal_plot)
    ax_first.set_xlim(0, DISPLAYED_TIME)
    ax_first.set_ylim(-1.0, 1.0)
    ax_first.set_xlabel("Time (s)")
    ax_first.set_ylabel("Amplitude")
    ax_first.set_title("Waveform")

    # Time plot - second input
    time_second, = ax_second.plot(t, second_signal_plot)
    ax_second.set_xlim(0, DISPLAYED_TIME)
    ax_second.set_ylim(-1.0, 1.0)
    ax_second.set_xlabel("Time (s)")
    ax_second.set_ylabel("Amplitude")
    ax_second.set_title("Waveform")

    # Buffers for saving
    recorded_blocks_first= []
    recorded_blocks_second=[]
    angles = []
    begin_time_angle = []
    end_time_angle = []
    processed_blocks = 0

    def update(_frame):
        nonlocal processed_blocks
        nonlocal first_signal_plot
        nonlocal second_signal_plot
        # If data is available, take the latest block
        if not audio_q.empty():
            block = audio_q.get()
            first_audio,second_audio = block[:,0],block[:,1]
            processed_blocks += 1
            # if detect_interest_noise(first_audio, second_audio):
            #     angle = calculate_angle(first_audio, second_audio)
            #     print(f"Measured angle: {angle}")
                #if SAVE_AUDIO:
                    #     angles.append(angle)
                    #     begin_time_angle.append((processed_blocks-1)*BLOCK_TIME)
                    #     end_time_angle.append((processed_blocks)*BLOCK_TIME)
            first_signal_plot,second_signal_plot = np.roll(first_signal_plot, -BLOCK_SIZE),np.roll(second_signal_plot,-BLOCK_SIZE)
            first_signal_plot[-BLOCK_SIZE:]=first_audio
            second_signal_plot[-BLOCK_SIZE:]=second_audio

            # Update time-domain data
            time_first.set_ydata(first_signal_plot)
            time_second.set_ydata(second_signal_plot)

            # Cache data for file outputs
            if SAVE_AUDIO:
                recorded_blocks_first.append(first_audio.astype(np.float32))
                recorded_blocks_second.append(second_audio.astype(np.float32))
                
        return time_first, time_second

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
    if SAVE_AUDIO and recorded_blocks_first:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_first = np.concatenate(recorded_blocks_first)
        audio_second = np.concatenate(recorded_blocks_second)
        wav_path_first = os.path.join(OUTPUT_DIR, f"mic_audio_{stamp}_first.wav")
        wav_path_second = os.path.join(OUTPUT_DIR, f"mic_audio_{stamp}_second.wav")
        save_wav(wav_path_first, audio_first, SAMPLE_RATE)
        save_wav(wav_path_second,audio_second, SAMPLE_RATE)
    if SAVE_ANGLES and angles:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUTPUT_DIR,f"angles_saved_{stamp}.csv")
        save_angle(path=path,angles=angles,begin_time=begin_time_angle,end_time=end_time_angle)

if __name__ == "__main__":
    print("Starting real-time microphone recorder.")
    print("Close the plot window to exit.")
    print("Default device setting:", sd.default.device)
    main()