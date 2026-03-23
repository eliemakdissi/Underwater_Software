import argparse
from datetime import datetime
from pathlib import Path

from utils.audio_pipeline import (
    TARGET_FREQS_HZ,
    apply_multi_band_filter,
    find_input_wav,
    process_audio,
    read_wav_mono,
    save_analysis_outputs,
)


def main():
    script_dir = Path(__file__).resolve().parent
    default_input_dir = script_dir / "input"
    default_output_root = script_dir / "output"

    parser = argparse.ArgumentParser(
        description="Analyze one WAV from input folder and write all results to output folder."
    )
    parser.add_argument(
        "--input-dir",
        default=str(default_input_dir),
        help="Folder containing WAV files (default: ./input).",
    )
    parser.add_argument(
        "--wav",
        default=None,
        help="Specific WAV filename or full path. If omitted, first WAV in input folder is used.",
    )
    parser.add_argument(
        "--output-root",
        default=str(default_output_root),
        help="Root output folder where a timestamped run directory is created (default: ./output).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=8192,
        help="FFT block size in samples (default: 8192).",
    )
    parser.add_argument(
        "--relative-margin",
        type=float,
        default=0.03,
        help="Relative half-bandwidth around 8.8k/37.5k bands (default: 0.03 => +/-3%%).",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=6,
        help="Butterworth order for each bandpass filter (default: 6).",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "double"),
        default="double",
        help="Filtering mode: single keeps one frequency, double keeps two frequencies (default: double).",
    )
    parser.add_argument(
        "--single-frequency",
        type=float,
        default=TARGET_FREQS_HZ[0],
        help=f"Frequency in Hz used when --mode single (default: {TARGET_FREQS_HZ[0]}).",
    )
    parser.add_argument(
        "--band-gain",
        type=float,
        default=20.0,
        help="Gain applied after summing filtered bands (default: 20.0).",
    )
    parser.add_argument(
        "--overlap-rate",
        type=float,
        default = 0.7,
        help="Overlap rate to perform FFT",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    block_size = args.block_size
    if block_size <= 0:
        raise ValueError("--block-size must be a positive integer.")
    if args.relative_margin <= 0:
        raise ValueError("--relative-margin must be positive.")
    if args.filter_order <= 0:
        raise ValueError("--filter-order must be a positive integer.")
    if args.single_frequency <= 0:
        raise ValueError("--single-frequency must be positive.")
    if args.band_gain <= 0:
        raise ValueError("--band-gain must be positive.")
    if args.overlap_rate <=0:
        raise ValueError("--overlap-rate must be between 0 and 1.")

    if args.wav:
        candidate = Path(args.wav)
        wav_path = candidate.resolve() if candidate.is_absolute() else (input_dir / candidate).resolve()
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")
    else:
        wav_path = find_input_wav(input_dir)

    sample_rate, mono_audio, channels = read_wav_mono(wav_path)
    target_freqs = [args.single_frequency] if args.mode == "single" else list(TARGET_FREQS_HZ)
    filtered_audio = apply_multi_band_filter(
        mono_audio,
        sample_rate,
        target_freqs,
        args.relative_margin,
        args.filter_order,
        args.band_gain,
    )
    processed = process_audio(filtered_audio, sample_rate, block_size)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{wav_path.stem}_{stamp}"
    run_output_dir = output_root / run_name

    save_analysis_outputs(
        wav_path,
        run_output_dir,
        sample_rate,
        channels,
        filtered_audio,
        processed,
        target_freqs,
        args.relative_margin,
        args.filter_order,
        args.band_gain,
    )

    print("Analysis complete.")
    print(f"Input WAV: {wav_path}")
    print(f"Mode: {args.mode}")
    print(f"Filtered frequencies: {', '.join(f'{freq:.1f} Hz' for freq in target_freqs)}")
    print(f"Band gain: x{args.band_gain:g}")
    print(f"Results folder: {run_output_dir}")


if __name__ == "__main__":
    main()
