# Filter Design and Implementation

This document explains how filtering is implemented in this project for offline processing in `main.py` through `utils/audio_pipeline.py`.

## Goal

Keep only acoustic energy around two beacon frequencies:
- `8.8 kHz`
- `37.5 kHz`

All other frequency content is attenuated as much as possible.

## Where the Filtering Happens

The filtering pipeline is implemented in:
- `utils/audio_pipeline.py` in `selective_bandpass(...)`
- `utils/audio_pipeline.py` in `apply_multi_band_filter(...)`

`main.py` calls these functions before any FFT analysis, so all downstream outputs are based on the filtered signal.

## Processing Steps

1. Read WAV and convert to mono float (`[-1, 1]`).
2. Build one narrow band-pass filter for each target frequency selected by the mode.
3. Apply all selected filters independently to the same signal.
4. Sum filtered outputs to form the final filtered waveform.
5. Apply band gain (`--band-gain`, default `20`).
6. Normalize only if peak amplitude exceeds `1.0` (prevents clipping when saving WAV).

### How One Narrow Band-Pass Filter Is Built

For one target frequency `f0` (for example `8800 Hz`), the code builds a digital filter in three steps:

1. Compute the band edges from the relative margin:
   - `lowcut = f0 * (1 - relative_margin)`
   - `highcut = f0 * (1 + relative_margin)`
2. Use SciPy to design an IIR Butterworth band-pass in SOS format:
   - `signal.butter(order, [lowcut, highcut], btype="bandpass", fs=sample_rate, output="sos")`
3. Apply it with zero-phase filtering:
   - `signal.sosfiltfilt(sos, data)`

Practical meaning of the parameters:
- `relative_margin`: controls bandwidth (smaller = narrower band).
- `order`: controls roll-off sharpness (higher = steeper transition).
- `fs=sample_rate`: tells SciPy the frequency scale in Hz directly.
- `output="sos"`: improves numerical stability versus direct-form coefficients.

Example with defaults (`relative_margin=0.03`, `f0=8800`):
- `lowcut = 8800 * 0.97 = 8536 Hz`
- `highcut = 8800 * 1.03 = 9064 Hz`

So only energy near `8.8 kHz` is kept, while frequencies outside this window are attenuated.

## Filter Type

Each band uses a Butterworth band-pass filter (`scipy.signal.butter`) in SOS form:
- **Type**: IIR Butterworth
- **Implementation**: second-order sections (`output="sos"`)
- **Application**: zero-phase forward-backward filtering (`scipy.signal.sosfiltfilt`)

Why this choice:
- Butterworth gives a smooth monotonic passband.
- SOS is numerically stable for higher orders.
- `sosfiltfilt` removes phase distortion in offline analysis.

## Band Definition

For each center frequency `f0`, the band edges are:
- `lowcut = f0 * (1 - relative_margin)`
- `highcut = f0 * (1 + relative_margin)`

Default values in `main.py`:
- `relative_margin = 0.03` (about +/-3%)
- `filter_order = 6`

That means approximate passbands:
- Around `8.8 kHz`: `8536 Hz` to `9064 Hz`
- Around `37.5 kHz`: `36375 Hz` to `38625 Hz`

## Safety Check (Nyquist)

Before designing each filter, the code checks:
- `highcut < sample_rate / 2`

If not, it raises an error because no valid digital band-pass can exist above Nyquist.

## Runtime Tuning

You can tune filtering behavior from `main.py`:

```powershell
python main.py --relative-margin 0.02 --filter-order 8 --band-gain 20
```

- Smaller `relative_margin` => narrower passband, stronger rejection, more sensitivity to frequency drift.
- Larger `filter_order` => steeper roll-off, but higher computational cost.

## Single vs Double Mode

`main.py` supports two filtering modes:

- `--mode double`: keep both target frequencies (`8800 Hz` and `37500 Hz`).
- `--mode single`: keep only one frequency defined by `--single-frequency`.

Examples:

```powershell
# Double mode (default): keep 8.8kHz and 37.5kHz
python main.py --mode double

# Single mode: keep only 8.8kHz
python main.py --mode single --single-frequency 8800

# Single mode: keep only 37.5kHz
python main.py --mode single --single-frequency 37500
```

## Output Impact

After filtering, the following outputs are generated from the filtered signal:
- `filtered_signal.wav`
- `spectrum.npz`
- `dominant_freq.csv`
- `analysis_plots.png`
- `metadata.json`
