from __future__ import annotations

from queue import Queue

import numpy as np

HOP_RATIOS = {"1/1": 1.0, "1/2": 0.5, "1/4": 0.25, "1/8": 0.125}


def hop_ratio_from_text(hop_text: str, default: float = 0.5) -> float:
    return HOP_RATIOS.get(hop_text, default)


def compute_hop_size(fft_size: int, hop_text: str) -> int:
    return int(int(fft_size) * hop_ratio_from_text(hop_text))


def compute_spectrogram_width(duration_s: int, sample_rate: int, hop_size: int) -> int:
    return max(1, int((int(duration_s) * int(sample_rate)) / int(hop_size)))


def drain_audio_queue(audio_queue: Queue) -> np.ndarray | None:
    chunks: list[np.ndarray] = []
    while not audio_queue.empty():
        chunks.append(audio_queue.get())
    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32, copy=False)


def update_waveform_history(history: np.ndarray, samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return history
    crop = samples[-len(history) :] if len(samples) > len(history) else samples
    updated = np.roll(history, -len(crop))
    updated[-len(crop) :] = crop
    return updated.astype(np.float32, copy=False)


def incremental_fft_db_columns(
    overlap_buffer: np.ndarray,
    new_samples: np.ndarray,
    fft_size: int,
    hop_size: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    audio_to_process = np.concatenate([overlap_buffer, new_samples]).astype(np.float32, copy=False)
    window = np.hanning(int(fft_size)).astype(np.float32)
    cols: list[np.ndarray] = []
    idx = 0

    while idx + fft_size <= len(audio_to_process):
        chunk = audio_to_process[idx : idx + fft_size]
        fft_complex = np.fft.rfft(chunk * window) / float(fft_size)
        cols.append((20 * np.log10(np.abs(fft_complex) + 1e-9)).astype(np.float32, copy=False))
        idx += int(hop_size)

    new_overlap = audio_to_process[idx:].astype(np.float32, copy=False)
    if not cols:
        return None, new_overlap
    return np.column_stack(cols).astype(np.float32, copy=False), new_overlap


def update_spectrogram_history(
    history: np.ndarray | None,
    new_data: np.ndarray,
    img_width: int,
    fill_value: float = -100.0,
) -> np.ndarray:
    n_cols = new_data.shape[1]
    num_freq_bins = new_data.shape[0]

    if (
        history is None
        or history.shape[0] != num_freq_bins
        or history.shape[1] != img_width
    ):
        history = np.full((num_freq_bins, img_width), fill_value, dtype=np.float32)

    history = np.roll(history, -n_cols, axis=1)
    history[:, -n_cols:] = new_data
    return history


def mirror_input_to_output_channels(input_block: np.ndarray, outdata: np.ndarray) -> None:
    if input_block.shape[1] == 1 and outdata.shape[1] == 2:
        outdata[:, 0] = input_block[:, 0]
        outdata[:, 1] = input_block[:, 0]
        return
    if input_block.shape[1] == outdata.shape[1]:
        outdata[:] = input_block
        return
    for ch in range(outdata.shape[1]):
        outdata[:, ch] = input_block[:, 0]
