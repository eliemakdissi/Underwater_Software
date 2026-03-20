# Underwater Acoustic Signal Analysis

This repository processes underwater audio recordings, keeps only the target beacon bands (`8.8 kHz` and `37.5 kHz`), and generates analysis artifacts (filtered WAV, spectrum, dominant frequency track, plots, and metadata).

## 1) Pull This Branch

From PowerShell:

```powershell
cd E:\MINES\UNDERWATER
git clone <repo-url> acoustique
cd acoustique
git fetch origin
git checkout <branch-name>
git pull origin <branch-name>
```

If you already cloned the repo:

```powershell
cd E:\MINES\UNDERWATER\acoustique
git fetch origin
git checkout <branch-name>
git pull origin <branch-name>
```

## 2) Install Dependencies

Use Python 3.10+ (recommended).

```powershell
cd E:\MINES\UNDERWATER\acoustique
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pandas sounddevice
```

Notes:
- `main.py` does **not** require microphone hardware.
- Real-time scripts (`realtime_*.py`) require a working input audio device and `sounddevice`.

## 3) Main Workflow (`main.py`)

`main.py` is the primary entrypoint for offline processing.
For implementation details of the dual-band filtering stage, see `filter.md`.

What it does:
- Reads one WAV from `input/` (or a specific file you pass).
- Applies either single-band or dual-band filtering, depending on mode.
- Analyzes the **filtered** signal with frame-wise FFT.
- Creates a timestamped result folder under `output/`.

Run with defaults:

```powershell
cd E:\MINES\UNDERWATER\acoustique
python main.py
```

Run with explicit arguments:

```powershell
python main.py `
  --input-dir "E:\MINES\UNDERWATER\acoustique\input" `
  --wav "clean_record_sea.wav" `
  --output-root "E:\MINES\UNDERWATER\acoustique\output" `
  --block-size 8192 `
  --relative-margin 0.03 `
  --filter-order 6 `
  --mode double `
  --band-gain 20
```

### Filtering Modes

- `double` mode: keep both target frequencies (`8.8 kHz` and `37.5 kHz`).
- `single` mode: keep only one frequency set by `--single-frequency`.

Examples:

```powershell
# Keep both 8.8kHz and 37.5kHz (default behavior)
python main.py --mode double

# Keep only 8.8kHz
python main.py --mode single --single-frequency 8800

# Keep only 37.5kHz
python main.py --mode single --single-frequency 37500
```

### Output Files Per Run

Inside `output/<input_stem>_<timestamp>/`:
- `filtered_signal.wav`: filtered audio containing only the two target bands.
- `spectrum.npz`: saved FFT/spectrogram matrix and frame metadata.
- `dominant_freq.csv`: dominant frequency over time.
- `analysis_plots.png`: waveform + spectrogram + dominant-frequency figure.
- `metadata.json`: processing configuration and file summary.

## 4) File-by-File Overview (One Sentence Each)

- `main.py`: Minimal CLI entrypoint that filters and analyzes one WAV file end-to-end.
- `filter.md`: Technical note describing how the dual-band filtering is designed and applied.
- `plot_recorded_outputs.py`: Utility script to visualize saved WAV/NPZ/CSV outputs from previous recordings.
- `realtime_mic_spectrum.py`: Real-time single-microphone recorder and spectrum analyzer with optional output saving.
- `realtime_double_mic.py`: Real-time dual-microphone viewer/recorder prepared for angle-estimation workflows.
- `test_fft.py`: Lightweight FFT/WAV loading test script for quick local checks.
- `mp3_spectre.ipynb`: Notebook for exploratory spectrum analysis experiments.
- `utils/__init__.py`: Package export file exposing shared legacy utility functions.
- `utils/legacy.py`: Shared low-level helpers for WAV saving, angle logic, and simple band-pass filtering.
- `utils/audio_pipeline.py`: Core offline pipeline functions used by `main.py` (read/filter/analyze/save).
- `input/clean_record_sea.wav`: Example input recording used by default commands.
- `output/`: Generated result folders created by processing scripts.

## 5) Typical End-to-End Example

```powershell
cd E:\MINES\UNDERWATER\acoustique
python main.py --wav clean_record_sea.wav
```

Then open the newest folder under `output/` and inspect `analysis_plots.png` and `dominant_freq.csv`.
