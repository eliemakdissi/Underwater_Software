from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal


def design_bandpass_sos(
    sample_rate: float,
    target_freq: float,
    relative_margin: float = 0.03,
    order: int = 6,
) -> np.ndarray:
    """Design a Butterworth band-pass filter in SOS form for realtime use."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if target_freq <= 0:
        raise ValueError("target_freq must be positive.")
    if relative_margin <= 0:
        raise ValueError("relative_margin must be positive.")
    if order <= 0:
        raise ValueError("order must be a positive integer.")

    lowcut = target_freq * (1.0 - relative_margin)
    highcut = target_freq * (1.0 + relative_margin)
    nyquist = sample_rate / 2.0

    if lowcut <= 0:
        raise ValueError("Computed lowcut must be > 0. Lower target_freq or margin.")
    if highcut >= nyquist:
        raise ValueError(
            f"Target {target_freq} Hz with margin {relative_margin} exceeds Nyquist ({nyquist} Hz)."
        )
    if lowcut >= highcut:
        raise ValueError("Computed lowcut must be strictly lower than highcut.")

    sos = signal.butter(
        int(order),
        [lowcut, highcut],
        btype="bandpass",
        fs=float(sample_rate),
        output="sos",
    )
    return np.asarray(sos, dtype=np.float64)


@dataclass
class StreamingBandpassFilter:
    """Stateful SOS band-pass filter for block-by-block realtime audio."""

    sos: np.ndarray
    _zi_template: np.ndarray = field(init=False, repr=False)
    zi: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.sos = np.asarray(self.sos, dtype=np.float64)
        self._zi_template = signal.sosfilt_zi(self.sos).astype(np.float64, copy=False)
        self.reset()

    @classmethod
    def from_frequency(
        cls,
        sample_rate: float,
        target_freq: float,
        relative_margin: float = 0.03,
        order: int = 6,
    ) -> "StreamingBandpassFilter":
        sos = design_bandpass_sos(
            sample_rate=sample_rate,
            target_freq=target_freq,
            relative_margin=relative_margin,
            order=order,
        )
        return cls(sos=sos)

    def reset(self) -> None:
        self.zi = self._zi_template.copy()

    def process_block(self, block: np.ndarray) -> np.ndarray:
        """Filter one mono block and preserve internal state."""
        mono = np.asarray(block, dtype=np.float32)
        if mono.ndim != 1:
            raise ValueError("process_block expects a 1D mono block.")
        if mono.size == 0:
            return mono.copy()

        filtered, self.zi = signal.sosfilt(self.sos, mono, zi=self.zi)
        return np.asarray(filtered, dtype=np.float32)


def process_multichannel_block(
    block: np.ndarray,
    filters: list[StreamingBandpassFilter],
) -> np.ndarray:
    """Filter each channel with its own stateful filter instance."""
    channels = np.asarray(block, dtype=np.float32)
    if channels.ndim != 2:
        raise ValueError("process_multichannel_block expects shape (samples, channels).")
    if channels.shape[1] != len(filters):
        raise ValueError("Number of filters must match number of channels.")

    out = np.empty_like(channels, dtype=np.float32)
    for ch_idx, channel_filter in enumerate(filters):
        out[:, ch_idx] = channel_filter.process_block(channels[:, ch_idx])
    return out
