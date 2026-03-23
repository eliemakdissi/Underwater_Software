# Underwater Acoustic Signal Analysis

This repository processes underwater audio recordings, keeps selected beacon bands (single or double mode), and generates analysis artifacts (filtered WAV, spectrum, dominant frequency track, plots, and metadata).

## 1) Clone / Pull the `acoustique` Branch

Reference branch:
- [Underwater_Software - acoustique](https://github.com/eliemakdissi/Underwater_Software/tree/acoustique)

### First-time clone

```powershell
cd E:\
git clone --branch acoustique --single-branch https://github.com/eliemakdissi/Underwater_Software.git
cd Underwater_Software
```

### Update an existing clone

```powershell
cd E:\Underwater_Software
git fetch origin
git checkout acoustique
git pull --ff-only origin acoustique
```

### Verify current branch

```powershell
git branch --show-current
```

Expected output: `acoustique`.

## 2) Install Dependencies

Use Python 3.10+ (recommended):

```powershell
cd E:\Underwater_Software
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pandas sounddevice
```

Notes:
- `main.py` does **not** require microphone hardware.
- Real-time scripts (`realtime_*.py`) require a working input audio device and `sounddevice`.

## 3) Quick Start

```powershell
cd E:\Underwater_Software
python main.py --mode double
```

Then open the newest folder under `output\` and inspect:
- `analysis_plots.png`
- `dominant_freq.csv`
- `metadata.json`

## 4) Main Workflow (`main.py`)

`main.py` is the primary offline entrypoint.  
For filter implementation details, see [filter.md](filter.md).

What it does:
- Reads one WAV from `input/` (or a specific file you pass).
- Applies `single` or `double` band-pass filtering.
- Amplifies selected band content with `--band-gain` (default `20`).
- Analyzes the filtered signal with frame-wise FFT.
- Creates a timestamped output folder under `output/`.

Run with defaults:

```powershell
cd E:\Underwater_Software
python main.py
```

Run with explicit arguments:

```powershell
python main.py `
  --input-dir "E:\Underwater_Software\input" `
  --wav "clean_record_sea.wav" `
  --output-root "E:\Underwater_Software\output" `
  --block-size 8192 `
  --relative-margin 0.03 `
  --filter-order 6 `
  --mode double `
  --band-gain 20
```

### Filtering Modes

- `double`: keep both `8.8 kHz` and `37.5 kHz`.
- `single`: keep one frequency via `--single-frequency`.

Examples:

```powershell
# Double mode (default): keep 8.8kHz + 37.5kHz
python main.py --mode double

# Single mode: keep only 8.8kHz
python main.py --mode single --single-frequency 8800

# Single mode: keep only 37.5kHz
python main.py --mode single --single-frequency 37500
```

### Output Files Per Run

Inside `output/<input_stem>_<timestamp>/`:
- `filtered_signal.wav`: filtered audio after mode selection and gain.
- `spectrum.npz`: frame-wise FFT/spectrogram matrix and metadata.
- `dominant_freq.csv`: dominant frequency over time.
- `analysis_plots.png`: waveform + spectrogram + dominant-frequency plot.
- `metadata.json`: configuration and run summary.

## 5) File-by-File Overview (One Sentence Each)

- `main.py`: Minimal CLI entrypoint that filters and analyzes one WAV file end-to-end.
- `filter.md`: Technical note describing filter design and usage (`single`/`double` modes).
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

## 6) Typical End-to-End Example

```powershell
cd E:\Underwater_Software
python main.py --wav clean_record_sea.wav --mode double --band-gain 20
```

Then open the newest folder under `output/` and inspect `analysis_plots.png` and `dominant_freq.csv`.
