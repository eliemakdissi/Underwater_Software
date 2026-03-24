from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import wiener

from .realtime_filters import StreamingBandpassFilter, design_bandpass_sos


@dataclass
class BandpassWienerProcessor:
    """Stateful realtime processor: narrow band-pass + Wiener denoise."""

    sample_rate: float
    target_freq: float
    bandwidth_hz: float
    order: int = 4
    wiener_window: int = 31
    _sos: np.ndarray = field(init=False, repr=False)
    _channel_filters: list[StreamingBandpassFilter] = field(
        init=False, default_factory=list, repr=False
    )

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.target_freq <= 0:
            raise ValueError("target_freq must be positive.")
        if self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive.")
        if self.order <= 0:
            raise ValueError("order must be a positive integer.")

        relative_margin = self.bandwidth_hz / (2.0 * self.target_freq)
        self._sos = design_bandpass_sos(
            sample_rate=self.sample_rate,
            target_freq=self.target_freq,
            relative_margin=relative_margin,
            order=self.order,
        )

    def reset(self) -> None:
        self._channel_filters = []

    def _ensure_channel_filters(self, channels: int) -> None:
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if len(self._channel_filters) == channels:
            return
        self._channel_filters = [StreamingBandpassFilter(self._sos) for _ in range(channels)]

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process one block shaped (samples, channels)."""
        block = np.asarray(data, dtype=np.float32)
        if block.ndim != 2:
            raise ValueError("process expects shape (samples, channels).")
        if block.size == 0:
            return block.copy()

        self._ensure_channel_filters(block.shape[1])

        filtered = np.empty_like(block, dtype=np.float32)
        win_size = min(int(self.wiener_window), block.shape[0])
        if win_size % 2 == 0:
            win_size = max(1, win_size - 1)

        for ch_idx, channel_filter in enumerate(self._channel_filters):
            bandpassed = channel_filter.process_block(block[:, ch_idx])
            filtered[:, ch_idx] = wiener(bandpassed, mysize=win_size).astype(
                np.float32, copy=False
            )

        return filtered
