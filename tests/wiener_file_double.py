import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

# Allow direct script execution from tests/ while still importing project packages.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from utils.legacy import calculate_angle, correlation
from utils.realtime_export import (
    create_run_folder,
    export_spectrogram_png,
    export_waveform_png,
)
from utils.realtime_stream import (
    compute_hop_size,
    compute_spectrogram_width,
    drain_audio_queue,
    incremental_fft_db_columns,
    update_spectrogram_history,
    update_waveform_history,
)


SAMPLE_RATE = 192000
TARGET_FREQS_HZ = (8800.0, 37500.0)
TARGET_BANDWIDTH_HZ = 1000.0
FILTER_ORDER = 4
WIENER_WINDOW = 31
INPUT_CHANNELS = 2
DISTANCE_HYDROPHONES = 0.62
CELERITY = 1500
GUI_REFRESH = 50  # Increased from 150ms to match audio block rate (~42.7ms per block at 192kHz/8192 blocksize)
NUM_BLOCKS_FOR_ANGLE_CALCULATION = 24
ANGLE_BLOCK_SIZE = 4096


@dataclass
class MultiBandDoubleWienerProcessor:
    """Realtime processor: multi-band band-pass + two Wiener passes."""

    sample_rate: float
    target_freqs: tuple[float, ...]
    bandwidth_hz: float
    order: int = 4
    wiener_window: int = 31
    _sos_filters: list[np.ndarray] = field(init=False, default_factory=list, repr=False)
    _channel_zi: list[list[np.ndarray]] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if not self.target_freqs:
            raise ValueError("target_freqs must contain at least one frequency.")
        if self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive.")
        if self.order <= 0:
            raise ValueError("order must be a positive integer.")

        nyquist = self.sample_rate / 2.0
        half_band = self.bandwidth_hz / 2.0
        self._sos_filters = []
        for target_freq in self.target_freqs:
            if target_freq <= 0:
                raise ValueError("All target frequencies must be positive.")
            lowcut = float(target_freq) - half_band
            highcut = float(target_freq) + half_band
            if lowcut <= 0 or highcut >= nyquist:
                raise ValueError(
                    "Invalid band: each target +/- bandwidth_hz/2 must stay within (0, Nyquist)."
                )
            sos = signal.butter(
                self.order,
                [lowcut, highcut],
                btype="bandpass",
                fs=self.sample_rate,
                output="sos",
            ).astype(np.float32)
            self._sos_filters.append(sos)

    def reset(self) -> None:
        self._channel_zi = []

    def _ensure_channel_state(self, channels: int) -> None:
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if len(self._channel_zi) == channels:
            return
        self._channel_zi = []
        for _ in range(channels):
            channel_states = []
            for sos in self._sos_filters:
                channel_states.append(signal.sosfilt_zi(sos).astype(np.float32))
            self._channel_zi.append(channel_states)

    def process(self, data: np.ndarray) -> np.ndarray:
        block = np.asarray(data, dtype=np.float32)
        if block.ndim != 2:
            raise ValueError("process expects shape (samples, channels).")
        if block.size == 0:
            return block.copy()

        self._ensure_channel_state(block.shape[1])
        filtered = np.empty_like(block, dtype=np.float32)
        win_size = min(int(self.wiener_window), block.shape[0])
        if win_size % 2 == 0:
            win_size = max(1, win_size - 1)

        for ch_idx in range(block.shape[1]):
            combined_band = np.zeros(block.shape[0], dtype=np.float32)
            for band_idx, sos in enumerate(self._sos_filters):
                bandpassed, self._channel_zi[ch_idx][band_idx] = signal.sosfilt(
                    sos, block[:, ch_idx], zi=self._channel_zi[ch_idx][band_idx]
                )
                combined_band += bandpassed.astype(np.float32, copy=False)
            first_pass = signal.wiener(combined_band, mysize=win_size)
            second_pass = signal.wiener(first_pass, mysize=win_size)
            filtered[:, ch_idx] = second_pass.astype(np.float32, copy=False)
        return filtered


class EnregistreurHydrophoneWienerDualFile(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hydrophone Analyzer from File (Dual Wiener)")
        self.samplerate = SAMPLE_RATE
        self.file_path = None
        self.audio_data = None
        self.current_sample_index = 0
        self.is_playing = False
        self.measured_angle = 0
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.output_root = os.path.join(project_root, "output")
        self.dossier_session = None

        self.filter_enabled = False
        self.filter_targets_hz = TARGET_FREQS_HZ
        self.filter_bandwidth_hz = TARGET_BANDWIDTH_HZ
        self.filter_processor = MultiBandDoubleWienerProcessor(
            sample_rate=self.samplerate,
            target_freqs=self.filter_targets_hz,
            bandwidth_hz=self.filter_bandwidth_hz,
            order=FILTER_ORDER,
            wiener_window=WIENER_WINDOW,
        )

        self.session_waveform_ch1 = []
        self.session_waveform_ch2 = []
        self.fft_size = 8192
        self.hop_size = 4096
        self.duree_visible = 5
        self.overlap_buffer_ch1 = np.array([], dtype=np.float32)
        self.overlap_buffer_ch2 = np.array([], dtype=np.float32)

        # For angle calculation queuing
        self.angle_buffer_ch1 = []
        self.angle_buffer_ch2 = []
        self.angle_buffer_start_sample = None  # Track where angle buffer started
        self.max_block_time = None  # Time of maximum amplitude block

        # Vertical lines for angle markers
        self.vline_spectro_ch1 = None
        self.vline_spectro_ch2 = None

        widget_central = QWidget()
        layout_principal = QVBoxLayout()
        pc = QFont("Arial", 10)

        layout_top = QHBoxLayout()
        layout_top.setSpacing(8)

        lbl_duree = QLabel("Hist(s):")
        lbl_duree.setFont(pc)
        self.combo_duree = QComboBox()
        self.combo_duree.setFont(pc)
        self.combo_duree.addItems(["1", "2", "5", "10", "15", "30", "60"])
        self.combo_duree.setCurrentText("5")
        self.combo_duree.currentTextChanged.connect(self.changer_duree)

        lbl_fft = QLabel("FFT:")
        lbl_fft.setFont(pc)
        self.combo_fft = QComboBox()
        self.combo_fft.setFont(pc)
        self.combo_fft.addItems(["1024", "2048", "4096", "8192", "16384"])
        self.combo_fft.setCurrentText("8192")
        self.combo_fft.currentTextChanged.connect(self.changer_parametres_calcul)

        lbl_hop = QLabel("Hop:")
        lbl_hop.setFont(pc)
        self.combo_hop = QComboBox()
        self.combo_hop.setFont(pc)
        self.combo_hop.addItems(["1/1", "1/2", "1/4", "1/8"])
        self.combo_hop.setCurrentText("1/2")
        self.combo_hop.currentTextChanged.connect(self.changer_parametres_calcul)

        lbl_canaux = QLabel("Channels: 2 (Dual Mic)")
        lbl_canaux.setFont(pc)

        lbl_info_filtre = QLabel(
            f"Targets: {self.filter_targets_hz[0]:.0f} Hz & {self.filter_targets_hz[1]:.0f} Hz "
            f"+/- {self.filter_bandwidth_hz/2:.0f} Hz | P=toggle Wiener"
        )
        lbl_info_filtre.setFont(pc)

        self.btn_charger = QPushButton("Load File")
        self.btn_charger.setFont(pc)
        self.btn_charger.setStyleSheet(
            "background-color:#1976D2;color:white;padding:6px;"
            "border-radius:4px;font-weight:bold;"
        )
        self.btn_charger.clicked.connect(self.charger_fichier)

        self.btn_demarrer = QPushButton("Play")
        self.btn_demarrer.setFont(pc)
        self.btn_demarrer.setStyleSheet(
            "background-color:#d32f2f;color:white;padding:6px;"
            "border-radius:4px;font-weight:bold;"
        )
        self.btn_demarrer.setEnabled(False)
        self.btn_demarrer.clicked.connect(self.demarrer_lecture)

        self.btn_arreter = QPushButton("Stop")
        self.btn_arreter.setFont(pc)
        self.btn_arreter.setStyleSheet(
            "background-color:#388E3C;color:white;padding:6px;"
            "border-radius:4px;font-weight:bold;"
        )
        self.btn_arreter.setEnabled(False)
        self.btn_arreter.clicked.connect(self.arreter_lecture)

        self.btn_quitter = QPushButton("Quit")
        self.btn_quitter.setFont(pc)
        self.btn_quitter.setStyleSheet(
            "background-color:#555;color:white;padding:6px;border-radius:4px;"
        )
        self.btn_quitter.clicked.connect(self.close)

        self.label_statut = QLabel("No file loaded")
        self.label_statut.setFont(QFont("Arial", 10, QFont.Bold))

        self.label_file = QLabel("")
        self.label_file.setFont(pc)

        for w in [
            lbl_duree,
            self.combo_duree,
            lbl_fft,
            self.combo_fft,
            lbl_hop,
            self.combo_hop,
            lbl_canaux,
        ]:
            layout_top.addWidget(w)

        layout_buttons = QHBoxLayout()
        for w in [self.btn_charger, self.btn_demarrer, self.btn_arreter, self.btn_quitter, self.label_statut]:
            layout_buttons.addWidget(w)

        self.lbl_measured_angle = QLabel(f"Measured angle: {self.measured_angle:.1f} deg")
        self.lbl_measured_angle.setFont(pc)

        layout_principal.addWidget(lbl_info_filtre)
        layout_principal.addLayout(layout_top)
        layout_principal.addLayout(layout_buttons)
        layout_principal.addWidget(self.label_file)
        layout_principal.addWidget(self.lbl_measured_angle)

        self.hist_fft_full_ch1 = None
        self.hist_fft_full_ch2 = None
        self.img_width = compute_spectrogram_width(
            duration_s=self.duree_visible,
            sample_rate=self.samplerate,
            hop_size=self.hop_size,
        )
        self.hist_onde_full_ch1 = np.zeros(self.duree_visible * self.samplerate, dtype=np.float32)
        self.hist_onde_full_ch2 = np.zeros(self.duree_visible * self.samplerate, dtype=np.float32)
        self.t_axis_onde = np.linspace(0, self.duree_visible, len(self.hist_onde_full_ch1), endpoint=False)

        self.win_graph = pg.GraphicsLayoutWidget()
        layout_principal.addWidget(self.win_graph)

        self.plot_onde_ch1 = self.win_graph.addPlot(title="Waveform CH1")
        self.plot_onde_ch1.setLabel("left", "Amplitude")
        self.plot_onde_ch1.setLabel("bottom", "Time", units="s")
        self.plot_onde_ch1.setYRange(-1.0, 1.0)
        self.plot_onde_ch1.setXRange(0, self.duree_visible, padding=0)
        self.plot_onde_ch1.showGrid(x=True, y=True, alpha=0.3)
        self.curve_onde_ch1 = self.plot_onde_ch1.plot(
            pen=pg.mkPen("b", width=2), autoDownsample=True, clipToView=True
        )

        self.win_graph.nextRow()

        self.plot_onde_ch2 = self.win_graph.addPlot(title="Waveform CH2")
        self.plot_onde_ch2.setLabel("left", "Amplitude")
        self.plot_onde_ch2.setLabel("bottom", "Time", units="s")
        self.plot_onde_ch2.setYRange(-1.0, 1.0)
        self.plot_onde_ch2.setXRange(0, self.duree_visible, padding=0)
        self.plot_onde_ch2.showGrid(x=True, y=True, alpha=0.3)
        self.curve_onde_ch2 = self.plot_onde_ch2.plot(
            pen=pg.mkPen("g", width=2), autoDownsample=True, clipToView=True
        )

        self.win_graph.nextRow()
        self.max_freq_khz = self.samplerate / 2000

        self.plot_spectro_ch1 = self.win_graph.addPlot(title="Spectrogram CH1")
        self.plot_spectro_ch1.setLabel("left", "Frequency", units="kHz")
        self.plot_spectro_ch1.setLabel("bottom", "Time", units="s")
        self.plot_spectro_ch1.setYRange(0, self.max_freq_khz, padding=0)
        self.plot_spectro_ch1.setXRange(0, self.duree_visible, padding=0)

        self.img_spectro_ch1 = pg.ImageItem()
        self.plot_spectro_ch1.addItem(self.img_spectro_ch1)

        cmap = pg.colormap.get("inferno")
        bar_ch1 = pg.ColorBarItem(values=(-100, -20), colorMap=cmap, label="Power (dB)")
        bar_ch1.setImageItem(self.img_spectro_ch1)

        self.win_graph.nextRow()

        self.plot_spectro_ch2 = self.win_graph.addPlot(title="Spectrogram CH2")
        self.plot_spectro_ch2.setLabel("left", "Frequency", units="kHz")
        self.plot_spectro_ch2.setLabel("bottom", "Time", units="s")
        self.plot_spectro_ch2.setYRange(0, self.max_freq_khz, padding=0)
        self.plot_spectro_ch2.setXRange(0, self.duree_visible, padding=0)

        self.img_spectro_ch2 = pg.ImageItem()
        self.plot_spectro_ch2.addItem(self.img_spectro_ch2)

        bar_ch2 = pg.ColorBarItem(values=(-100, -20), colorMap=cmap, label="Power (dB)")
        bar_ch2.setImageItem(self.img_spectro_ch2)

        layout_principal.setStretchFactor(self.win_graph, 1)
        widget_central.setLayout(layout_principal)
        self.setCentralWidget(widget_central)

        self.timer_gui = QTimer()
        self.timer_gui.timeout.connect(self.process_next_block)

    def charger_fichier(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Stereo WAV File", "", "WAV Files (*.wav)"
        )
        if not file_path:
            return

        try:
            self.samplerate, self.audio_data = wavfile.read(file_path)

            # Ensure data is float32 and stereo
            self.audio_data = self.audio_data.astype(np.float32, copy=False)
            if self.audio_data.ndim == 1:
                self.audio_data = np.column_stack([self.audio_data, self.audio_data])
            elif self.audio_data.shape[1] < 2:
                self.audio_data = np.column_stack([self.audio_data[:, 0], self.audio_data[:, 0]])

            self.file_path = file_path
            self.current_sample_index = 0
            self.is_playing = False

            duration = len(self.audio_data) / self.samplerate
            filename = os.path.basename(file_path)
            self.label_file.setText(f"File: {filename} | Duration: {duration:.1f}s | SR: {self.samplerate}Hz")

            self.label_statut.setText("File loaded")
            self.label_statut.setStyleSheet("color:#388E3C;")
            self.btn_demarrer.setEnabled(True)
        except Exception as e:
            self.label_statut.setText(f"Error: {e}")
            self.label_statut.setStyleSheet("color:#d32f2f;")
            print(e)

    def _reset_spectro(self):
        self.hist_fft_full_ch1 = None
        self.hist_fft_full_ch2 = None
        self.overlap_buffer_ch1 = np.array([], dtype=np.float32)
        self.overlap_buffer_ch2 = np.array([], dtype=np.float32)
        if hasattr(self, "img_spectro_ch1"):
            self.img_spectro_ch1.clear()
        if hasattr(self, "img_spectro_ch2"):
            self.img_spectro_ch2.clear()

    def _apply_optional_filter(self, indata):
        if not self.filter_enabled:
            return indata
        return self.filter_processor.process(indata)

    def _set_status(self, text, color):
        self.label_statut.setText(text)
        self.label_statut.setStyleSheet(f"color:{color};")

    def changer_duree(self, text):
        self.duree_visible = int(text)
        self.hist_onde_full_ch1 = np.zeros(self.duree_visible * self.samplerate, dtype=np.float32)
        self.hist_onde_full_ch2 = np.zeros(self.duree_visible * self.samplerate, dtype=np.float32)
        self.t_axis_onde = np.linspace(0, self.duree_visible, len(self.hist_onde_full_ch1), endpoint=False)
        self.img_width = compute_spectrogram_width(
            duration_s=self.duree_visible,
            sample_rate=self.samplerate,
            hop_size=self.hop_size,
        )
        self._reset_spectro()
        if hasattr(self, "plot_onde_ch1"):
            self.plot_onde_ch1.setXRange(0, self.duree_visible, padding=0)
        if hasattr(self, "plot_onde_ch2"):
            self.plot_onde_ch2.setXRange(0, self.duree_visible, padding=0)
        if hasattr(self, "plot_spectro_ch1"):
            self.plot_spectro_ch1.setXRange(0, self.duree_visible, padding=0)
        if hasattr(self, "plot_spectro_ch2"):
            self.plot_spectro_ch2.setXRange(0, self.duree_visible, padding=0)

    def changer_parametres_calcul(self):
        self.fft_size = int(self.combo_fft.currentText())
        self.hop_size = compute_hop_size(self.fft_size, self.combo_hop.currentText())
        self.img_width = compute_spectrogram_width(
            duration_s=self.duree_visible,
            sample_rate=self.samplerate,
            hop_size=self.hop_size,
        )
        self._reset_spectro()

    def update_angle(self):
        if len(self.angle_buffer_ch1) < NUM_BLOCKS_FOR_ANGLE_CALCULATION:
            return

        raw_ch1 = np.concatenate(self.angle_buffer_ch1[:NUM_BLOCKS_FOR_ANGLE_CALCULATION])
        raw_ch2 = np.concatenate(self.angle_buffer_ch2[:NUM_BLOCKS_FOR_ANGLE_CALCULATION])
        NUM_SAMPLES_TOT = self.fft_size*NUM_BLOCKS_FOR_ANGLE_CALCULATION
        time = np.linspace(0,NUM_SAMPLES_TOT/SAMPLE_RATE,NUM_SAMPLES_TOT)
        plt.plot(time,raw_ch1,label="Chanel1")
        plt.plot(time,raw_ch2,label="Chanel2")
        plt.savefig("raw_signals.png")
        plt.close()
        
        self.angle_buffer_ch1 = self.angle_buffer_ch1[NUM_BLOCKS_FOR_ANGLE_CALCULATION:]
        self.angle_buffer_ch2 = self.angle_buffer_ch2[NUM_BLOCKS_FOR_ANGLE_CALCULATION:]

        try:
            ch1_array = raw_ch1.reshape(-1, ANGLE_BLOCK_SIZE)
            ch2_array = raw_ch2.reshape(-1, ANGLE_BLOCK_SIZE)
            norm_ch1 = np.linalg.norm(ch1_array, axis=1)
            highest_index = np.argmax(norm_ch1)

            # Calculate time of the maximum amplitude block
            if self.angle_buffer_start_sample is not None:
                max_block_sample = self.angle_buffer_start_sample + highest_index * ANGLE_BLOCK_SIZE
                self.max_block_time = max_block_sample / self.samplerate
            



            ch1_relevant, ch2_relevant = ch1_array[highest_index], ch2_array[highest_index]
            time = np.linspace(0,ANGLE_BLOCK_SIZE/SAMPLE_RATE,ANGLE_BLOCK_SIZE)
            plt.plot(time,ch1_relevant,label="Chanel1")
            plt.plot(time,ch2_relevant,label="Chanel2")
            plt.savefig("interest_signals.png")
            plt.close()
            delay_samples = correlation(ch1_relevant, ch2_relevant)
            measured_angle = calculate_angle(
                delay_samples,
                sample_rate=self.samplerate,
                distance_microphones=DISTANCE_HYDROPHONES,
                celerity=CELERITY,
            )
            if measured_angle is not None:
                self.measured_angle = measured_angle
                print(f"Angle calculated: {self.measured_angle:.1f}° at t={self.max_block_time:.2f}s")
        except Exception as e:
            print(f"Error in angle calculation: {e}")


    def process_next_block(self):
        if not self.is_playing or self.audio_data is None:
            return

        # Get next block
        end_index = min(self.current_sample_index + self.fft_size, len(self.audio_data))
        block = self.audio_data[self.current_sample_index:end_index]

        if len(block) == 0:
            self.arreter_lecture()
            return

        # Pad if necessary
        if len(block) < self.fft_size:
            block = np.vstack([block, np.zeros((self.fft_size - len(block), 2), dtype=np.float32)])

        # Process block
        data_proc = self._apply_optional_filter(block)

        ch1 = -data_proc[:, 0].astype(np.float32, copy=True)
        ch2 = data_proc[:, 1].astype(np.float32, copy=True) if data_proc.shape[1] > 1 else ch1

        # Store for angle calculation
        if len(self.angle_buffer_ch1) == 0:
            self.angle_buffer_start_sample = self.current_sample_index

        self.angle_buffer_ch1.append(ch1.copy())
        self.angle_buffer_ch2.append(ch2.copy())
        if len(self.angle_buffer_ch1) >= NUM_BLOCKS_FOR_ANGLE_CALCULATION:
            self.update_angle()

        # Update waveform history
        self.hist_onde_full_ch1 = update_waveform_history(self.hist_onde_full_ch1, ch1)
        self.hist_onde_full_ch2 = update_waveform_history(self.hist_onde_full_ch2, ch2)

        self.lbl_measured_angle.setText(f"Measured angle: {self.measured_angle:.1f} deg")

        # Update waveform plots
        skip = max(1, len(self.hist_onde_full_ch1) // 4000)
        self.curve_onde_ch1.setData(x=self.t_axis_onde[::skip], y=self.hist_onde_full_ch1[::skip])
        self.curve_onde_ch2.setData(x=self.t_axis_onde[::skip], y=self.hist_onde_full_ch2[::skip])

        # Update spectrograms
        new_data_ch1, self.overlap_buffer_ch1 = incremental_fft_db_columns(
            overlap_buffer=self.overlap_buffer_ch1,
            new_samples=ch1,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
        )
        if new_data_ch1 is not None:
            self.hist_fft_full_ch1 = update_spectrogram_history(
                history=self.hist_fft_full_ch1,
                new_data=new_data_ch1,
                img_width=self.img_width,
                fill_value=-100.0,
            )
            self.img_spectro_ch1.setImage(
                self.hist_fft_full_ch1.T, autoLevels=False, levels=[-100, -20]
            )
            self.img_spectro_ch1.setRect(QRectF(0, 0, self.duree_visible, self.max_freq_khz))

        new_data_ch2, self.overlap_buffer_ch2 = incremental_fft_db_columns(
            overlap_buffer=self.overlap_buffer_ch2,
            new_samples=ch2,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
        )
        if new_data_ch2 is not None:
            self.hist_fft_full_ch2 = update_spectrogram_history(
                history=self.hist_fft_full_ch2,
                new_data=new_data_ch2,
                img_width=self.img_width,
                fill_value=-100.0,
            )
            self.img_spectro_ch2.setImage(
                self.hist_fft_full_ch2.T, autoLevels=False, levels=[-100, -20]
            )
            self.img_spectro_ch2.setRect(QRectF(0, 0, self.duree_visible, self.max_freq_khz))

        self.current_sample_index = end_index

    def demarrer_lecture(self):
        if self.audio_data is None:
            return

        self.current_sample_index = 0
        self.is_playing = True
        self.session_waveform_ch1 = []
        self.session_waveform_ch2 = []
        self.overlap_buffer_ch1 = np.array([], dtype=np.float32)
        self.overlap_buffer_ch2 = np.array([], dtype=np.float32)
        self.angle_buffer_ch1 = []
        self.angle_buffer_ch2 = []
        self.angle_buffer_start_sample = None
        self.max_block_time = None
        self.hist_fft_full_ch1 = None
        self.hist_fft_full_ch2 = None
        self.filter_processor.reset()
        self.hist_onde_full_ch1.fill(0)
        self.hist_onde_full_ch2.fill(0)

        # Clear any existing markers
        if self.vline_spectro_ch1 is not None:
            self.plot_spectro_ch1.removeItem(self.vline_spectro_ch1)
            self.vline_spectro_ch1 = None
        if self.vline_spectro_ch2 is not None:
            self.plot_spectro_ch2.removeItem(self.vline_spectro_ch2)
            self.vline_spectro_ch2 = None

        self.fft_size = int(self.combo_fft.currentText())
        self.hop_size = compute_hop_size(self.fft_size, self.combo_hop.currentText())
        self.img_width = compute_spectrogram_width(
            duration_s=self.duree_visible,
            sample_rate=self.samplerate,
            hop_size=self.hop_size,
        )

        self.timer_gui.start(GUI_REFRESH)

        filtre = "ON" if self.filter_enabled else "OFF"
        self._set_status(f"Playing | Wiener: {filtre}", "#d32f2f")
        self.btn_demarrer.setEnabled(False)
        self.btn_arreter.setEnabled(True)
        self.combo_duree.setEnabled(False)
        self.combo_fft.setEnabled(False)
        self.combo_hop.setEnabled(False)
        self.btn_charger.setEnabled(False)

    def arreter_lecture(self):
        self.timer_gui.stop()
        self.is_playing = False

        if self.audio_data is not None:
            self.dossier_session = create_run_folder(self.output_root, prefix="run")

            channel_1 = self.audio_data[:, 0]
            channel_2 = self.audio_data[:, 1] if self.audio_data.shape[1] > 1 else self.audio_data[:, 0]

            try:
                export_waveform_png(
                    channel_1,
                    self.samplerate,
                    self.dossier_session,
                    filename="waveform_ch1.png",
                    title="Waveform CH1 (full file)",
                )
                export_spectrogram_png(
                    channel_1,
                    self.samplerate,
                    self.fft_size,
                    self.hop_size,
                    self.dossier_session,
                    filename="spectrogram_ch1.png",
                    title="Spectrogram CH1 (full file)",
                )
                export_waveform_png(
                    channel_2,
                    self.samplerate,
                    self.dossier_session,
                    filename="waveform_ch2.png",
                    title="Waveform CH2 (full file)",
                )
                export_spectrogram_png(
                    channel_2,
                    self.samplerate,
                    self.fft_size,
                    self.hop_size,
                    self.dossier_session,
                    filename="spectrogram_ch2.png",
                    title="Spectrogram CH2 (full file)",
                )

                self._set_status(f"Analysis complete ({os.path.basename(self.dossier_session)})", "#388E3C")
            except Exception as export_error:
                self._set_status(f"Export error: {export_error}", "#d32f2f")
                print(f"Export error: {export_error}")

        self.btn_demarrer.setEnabled(True)
        self.btn_arreter.setEnabled(False)
        self.combo_duree.setEnabled(True)
        self.combo_fft.setEnabled(True)
        self.combo_hop.setEnabled(True)
        self.btn_charger.setEnabled(True)
        self.filter_processor.reset()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_P:
            self.filter_enabled = not self.filter_enabled
            self.filter_processor.reset()
            if self.is_playing:
                filtre = "ON" if self.filter_enabled else "OFF"
                self._set_status(f"Playing | Wiener: {filtre}", "#d32f2f")
            else:
                filtre = "ON" if self.filter_enabled else "OFF"
                self._set_status(f"Filter: {filtre}", "#1976D2")
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer_gui.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pg.setConfigOptions(useOpenGL=True, antialias=True, foreground="k", background="w")
    fenetre = EnregistreurHydrophoneWienerDualFile()
    fenetre.showMaximized()
    sys.exit(app.exec_())
