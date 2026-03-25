from __future__ import annotations

from typing import Iterable

import sounddevice as sd
from PyQt5.QtWidgets import QComboBox


def add_input_output_devices(
    combo_in: QComboBox,
    combo_out: QComboBox,
    devices: Iterable[dict] | None = None,
    max_name_len: int = 20,
) -> None:
    """Populate input/output device comboboxes with sounddevice entries."""
    source = sd.query_devices() if devices is None else devices
    for idx, device in enumerate(source):
        device_name = str(device.get("name", "unknown"))[:max_name_len]
        if device.get("max_input_channels", 0) > 0:
            combo_in.addItem(f"{idx}: {device_name}", idx)
        if device.get("max_output_channels", 0) > 0:
            combo_out.addItem(f"{idx}: {device_name}", idx)


def select_default_input_output_devices(
    combo_in: QComboBox,
    combo_out: QComboBox,
    default_device: tuple[int | None, int | None] | list[int | None] | None = None,
) -> None:
    """Select default input/output IDs when available."""
    try:
        defaults = default_device if default_device is not None else sd.default.device
        if defaults is None or len(defaults) < 2:
            return
        idx_in = combo_in.findData(defaults[0])
        idx_out = combo_out.findData(defaults[1])
        if idx_in >= 0:
            combo_in.setCurrentIndex(idx_in)
        if idx_out >= 0:
            combo_out.setCurrentIndex(idx_out)
    except Exception:
        # Keep UI usable even if default device info is unavailable.
        return
