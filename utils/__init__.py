from .legacy import (
    bandpass_filter,
    calculate_angle,
    detect_interest_noise,
    save_angle,
    save_wav,
)
from .realtime_filters import (
    StreamingBandpassFilter,
    design_bandpass_sos,
    process_multichannel_block,
)

__all__ = [
    "save_wav",
    "save_angle",
    "calculate_angle",
    "detect_interest_noise",
    "bandpass_filter",
    "design_bandpass_sos",
    "StreamingBandpassFilter",
    "process_multichannel_block",
]
