import os
import queue
import sys
from datetime import datetime

# Allow direct script execution from tests/ while still importing project packages.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.io import wavfile
from utils.audio_devices import (
    add_input_output_devices,
    select_default_input_output_devices,
)
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
    mirror_input_to_output_channels,
    update_spectrogram_history,
    update_waveform_history,
)
from utils.realtime_wiener import BandpassWienerProcessor
from utils.legacy import calculate_angle,correlation
from utils.timer import time_it


SAMPLE_RATE = 192000
TARGET_FREQ_HZ = 8800.0
TARGET_BANDWIDTH_HZ = 1000.0      
FILTER_ORDER = 4
WIENER_WINDOW = 31
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 1
DISTANCE_HYDROPHONES=1
CELERITY=340                       # Celerity of sound in the considered in the considered field
GUI_REFRESH = 150 


class EnregistreurHydrophoneWienerDualMic(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hydrophone Recorder (Dual Mic Wiener Test)")
        self.samplerate = SAMPLE_RATE
        self.audio_data_for_save = []
        self.audio_queue_ch1 = queue.Queue()
        self.audio_queue_ch2 = queue.Queue()
        self.stream = None
        self.measured_angle = 0
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.output_root = os.path.join(project_root, "output")
        self.dossier_session = None

        self.filter_enabled = False
        self.filter_center_hz = TARGET_FREQ_HZ
        self.filter_bandwidth_hz = TARGET_BANDWIDTH_HZ
        self.filter_processor = BandpassWienerProcessor(
            sample_rate=self.samplerate,
            target_freq=self.filter_center_hz,
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

        widget_central = QWidget()
        layout_principal = QVBoxLayout()
        pc = QFont("Arial", 10)

        layout_top = QHBoxLayout()
        layout_top.setSpacing(8)

        lbl_in = QLabel("In:")
        lbl_in.setFont(pc)
        self.combo_in = QComboBox()
        self.combo_in.setFont(pc)

        lbl_out = QLabel("Out:")
        lbl_out.setFont(pc)
        self.combo_out = QComboBox()
        self.combo_out.setFont(pc)

        add_input_output_devices(self.combo_in, self.combo_out)
        select_default_input_output_devices(self.combo_in, self.combo_out)

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

        lbl_canaux = QLabel("Canaux:")
        lbl_canaux.setFont(pc)
        self.combo_canaux = QComboBox()
        self.combo_canaux.setFont(pc)
        self.combo_canaux.addItems(["2 (Dual Mic)"])
        self.combo_canaux.setEnabled(False)

        lbl_info_filtre = QLabel(
            f"Target: {self.filter_center_hz:.0f} Hz +/- {self.filter_bandwidth_hz/2:.0f} Hz | P=toggle Wiener"
        )
        lbl_info_filtre.setFont(pc)

        for w in [
            lbl_in,
            self.combo_in,
            lbl_out,
            self.combo_out,
            lbl_duree,
            self.combo_duree,
            lbl_fft,
            self.combo_fft,
            lbl_hop,
            self.combo_hop,
            lbl_canaux,
            self.combo_canaux,
        ]:
            layout_top.addWidget(w)

        self.btn_demarrer = QPushButton("REC")
        self.btn_demarrer.setFont(pc)
        self.btn_demarrer.setStyleSheet(
            "background-color:#d32f2f;color:white;padding:6px;"
            "border-radius:4px;font-weight:bold;"
        )
        self.btn_demarrer.clicked.connect(self.demarrer_enregistrement)

        self.btn_arreter = QPushButton("STOP")
        self.btn_arreter.setFont(pc)
        self.btn_arreter.setStyleSheet(
            "background-color:#388E3C;color:white;padding:6px;"
            "border-radius:4px;font-weight:bold;"
        )
        self.btn_arreter.setEnabled(False)
        self.btn_arreter.clicked.connect(self.arreter_enregistrement)

        self.btn_quitter = QPushButton("Quitter")
        self.btn_quitter.setFont(pc)
        self.btn_quitter.setStyleSheet(
            "background-color:#555;color:white;padding:6px;border-radius:4px;"
        )
        self.btn_quitter.clicked.connect(self.close)

        self.label_statut = QLabel("Filter: OFF")
        self.label_statut.setFont(QFont("Arial", 10, QFont.Bold))

        for w in [self.btn_demarrer, self.btn_arreter, self.btn_quitter, self.label_statut]:
            layout_top.addWidget(w)

        self.lbl_measured_angle = QLabel(
            f"Measured angle: {self.measured_angle:.1f}°"
        )
        self.lbl_measured_angle.setFont(pc)

        layout_principal.addWidget(lbl_info_filtre)
        layout_principal.addLayout(layout_top)
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
        max_bin = 2000 
        max_freq_khz = (max_bin * self.samplerate / self.fft_size) / 1000.0

        self.plot_spectro_ch1 = self.win_graph.addPlot(title="Spectrogram CH1")
        self.plot_spectro_ch1.setLabel("left", "Frequency", units="kHz")
        self.plot_spectro_ch1.setLabel("bottom", "Time", units="s")
        self.plot_spectro_ch1.setYRange(0, max_freq_khz, padding=0)
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
        self.plot_spectro_ch2.setYRange(0, max_freq_khz, padding=0)
        self.plot_spectro_ch2.setXRange(0, self.duree_visible, padding=0)

        self.img_spectro_ch2 = pg.ImageItem()
        self.plot_spectro_ch2.addItem(self.img_spectro_ch2)

        bar_ch2 = pg.ColorBarItem(values=(-100, -20), colorMap=cmap, label="Power (dB)")
        bar_ch2.setImageItem(self.img_spectro_ch2)

        layout_principal.setStretchFactor(self.win_graph, 1)
        widget_central.setLayout(layout_principal)
        self.setCentralWidget(widget_central)

        self.timer_gui = QTimer()
        self.timer_gui.timeout.connect(self.update_charts)

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

    def _set_recording_status(self):
        filtre = "ON" if self.filter_enabled else "OFF"
        self.label_statut.setText(f"REC | Wiener: {filtre}")
        self.label_statut.setStyleSheet("color:#d32f2f;")

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

    @time_it
    def callback_audio(self, indata, outdata, frames, time, status):
        data_in = indata.astype(np.float32, copy=False)
        data_proc = self._apply_optional_filter(data_in)      # Dataprocessing through the filter

        self.audio_data_for_save.append(data_proc.copy())     # Saving the recorded audio
        if self.stream:
            ch1 = data_proc[:, 0].astype(np.float32, copy=True) 
            ch2 = data_proc[:, 1].astype(np.float32, copy=True) if data_proc.shape[1] > 1 else ch1
            self.audio_queue_ch1.put(ch1)
            self.audio_queue_ch2.put(ch2)
            self.session_waveform_ch1.append(ch1)
            self.session_waveform_ch2.append(ch2)

        mirror_input_to_output_channels(data_proc, outdata)   # It was commented before

    def update_charts(self):
        raw_ch1 = drain_audio_queue(self.audio_queue_ch1)
        raw_ch2 = drain_audio_queue(self.audio_queue_ch2)
        if raw_ch1 is None and raw_ch2 is None:
            return

        if raw_ch1 is None:
            raw_ch1 = raw_ch2
        if raw_ch2 is None:
            raw_ch2 = raw_ch1
        if raw_ch1 is None or raw_ch2 is None:
            return
        self.hist_onde_full_ch1 = update_waveform_history(self.hist_onde_full_ch1, raw_ch1)
        self.hist_onde_full_ch2 = update_waveform_history(self.hist_onde_full_ch2, raw_ch2)

        # Calcul de l'angle
        delay_samples = correlation(raw_ch1,raw_ch2)
        measured_angle=calculate_angle(delay_samples,
                                            sample_rate=self.samplerate,
                                            distance_microphones=DISTANCE_HYDROPHONES,
                                            celerity=CELERITY)
        if measured_angle!=None:
            self.measured_angle = measured_angle
            self.lbl_measured_angle.setText(f"Measured angle: {self.measured_angle:.1f}°")
            
        skip = max(1, len(self.hist_onde_full_ch1) // 4000)
        self.curve_onde_ch1.setData(x=self.t_axis_onde[::skip], y=self.hist_onde_full_ch1[::skip])
        self.curve_onde_ch2.setData(x=self.t_axis_onde[::skip], y=self.hist_onde_full_ch2[::skip])

        new_data_ch1, self.overlap_buffer_ch1 = incremental_fft_db_columns(
            overlap_buffer=self.overlap_buffer_ch1,
            new_samples=raw_ch1,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
        )
        max_bin = 2000 
        max_freq_khz = (max_bin * self.samplerate / self.fft_size) / 1000.0
        if new_data_ch1 is not None:

            self.hist_fft_full_ch1 = update_spectrogram_history(
                history=self.hist_fft_full_ch1,
                new_data=new_data_ch1,
                img_width=self.img_width,
                fill_value=-100.0,
            )
            light_image_ch1 = self.hist_fft_full_ch1[:max_bin,:]
            self.img_spectro_ch1.setImage(
                light_image_ch1.T, autoLevels=False, levels=[-100, -20]
            )
            self.img_spectro_ch1.setRect(QRectF(0, 0, self.duree_visible, max_freq_khz))

        new_data_ch2, self.overlap_buffer_ch2 = incremental_fft_db_columns(
            overlap_buffer=self.overlap_buffer_ch2,
            new_samples=raw_ch2,
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
            light_image_ch2 = self.hist_fft_full_ch2[:max_bin,:]
            self.img_spectro_ch2.setImage(
                light_image_ch2.T, autoLevels=False, levels=[-100, -20]
            )
            self.img_spectro_ch2.setRect(QRectF(0, 0, self.duree_visible, max_freq_khz))

    def demarrer_enregistrement(self):
        self.audio_data_for_save = []
        self.session_waveform_ch1 = []
        self.session_waveform_ch2 = []
        self.overlap_buffer_ch1 = np.array([], dtype=np.float32)
        self.overlap_buffer_ch2 = np.array([], dtype=np.float32)
        self.hist_fft_full_ch1 = None
        self.hist_fft_full_ch2 = None
        self.filter_processor.reset()
        self.hist_onde_full_ch1.fill(0)
        self.hist_onde_full_ch2.fill(0)

        self.dossier_session = create_run_folder(self.output_root, prefix="run")

        self.fft_size = int(self.combo_fft.currentText())
        self.hop_size = compute_hop_size(self.fft_size, self.combo_hop.currentText())
        self.img_width = compute_spectrogram_width(
            duration_s=self.duree_visible,
            sample_rate=self.samplerate,
            hop_size=self.hop_size,
        )

        idx_in = self.combo_in.currentData()
        idx_out = self.combo_out.currentData()

        try:
            input_info = sd.query_devices(idx_in)
            max_in = int(input_info.get("max_input_channels", 0))
            if max_in < INPUT_CHANNELS:
                raise ValueError(
                    f"Selected input supports only {max_in} channel(s); dual mic needs {INPUT_CHANNELS}."
                )

            self.stream = sd.Stream(
                device=(idx_in,idx_out),
                samplerate=self.samplerate,
                channels=(INPUT_CHANNELS, OUTPUT_CHANNELS),
                blocksize=self.fft_size,
                callback=self.callback_audio,
                dtype="float32",
            )
            self.stream.start()
            self.timer_gui.start(GUI_REFRESH)

            self._set_recording_status()
            self.btn_demarrer.setEnabled(False)
            self.btn_arreter.setEnabled(True)
            self.combo_duree.setEnabled(False)
            self.combo_in.setEnabled(False)
            self.combo_out.setEnabled(False)
        except Exception as e:
            self.label_statut.setText(f"Erreur: {e}")
            self.label_statut.setStyleSheet("color:#d32f2f;")
            print(e)

    def arreter_enregistrement(self):
        self.timer_gui.stop()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.label_statut.setText("Sauvegarde...")
        self.label_statut.setStyleSheet("color:#888;")
        QApplication.processEvents()

        if self.audio_data_for_save:
            enregistrement_complet = np.vstack(self.audio_data_for_save)
            horodatage = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
            nom_fichier = f"hydro_dual_{horodatage}.wav"
            dossier_cible = self.dossier_session or os.path.join(self.output_root, f"run_{horodatage}")
            os.makedirs(dossier_cible, exist_ok=True)

            wavfile.write(
                os.path.join(dossier_cible, nom_fichier),
                self.samplerate,
                enregistrement_complet,
            )

            try:
                channel_1 = enregistrement_complet[:, 0]
                channel_2 = (
                    enregistrement_complet[:, 1]
                    if enregistrement_complet.shape[1] > 1
                    else enregistrement_complet[:, 0]
                )
                export_waveform_png(
                    channel_1,
                    self.samplerate,
                    dossier_cible,
                    filename="waveform_ch1.png",
                    title="Waveform CH1 (full session)",
                )
                export_spectrogram_png(
                    channel_1,
                    self.samplerate,
                    self.fft_size,
                    self.hop_size,
                    dossier_cible,
                    filename="spectrogram_ch1.png",
                    title="Spectrogram CH1 (full session)",
                )
                export_waveform_png(
                    channel_2,
                    self.samplerate,
                    dossier_cible,
                    filename="waveform_ch2.png",
                    title="Waveform CH2 (full session)",
                )
                export_spectrogram_png(
                    channel_2,
                    self.samplerate,
                    self.fft_size,
                    self.hop_size,
                    dossier_cible,
                    filename="spectrogram_ch2.png",
                    title="Spectrogram CH2 (full session)",
                )
            except Exception as export_error:
                print(f"Erreur export figures: {export_error}")

            self.label_statut.setText(f"OK {nom_fichier} ({os.path.basename(dossier_cible)})")
            self.label_statut.setStyleSheet("color:#388E3C;")
        else:
            self.label_statut.setText("")

        self.btn_demarrer.setEnabled(True)
        self.btn_arreter.setEnabled(False)
        self.combo_duree.setEnabled(True)
        self.combo_in.setEnabled(True)
        self.combo_out.setEnabled(True)
        self.filter_processor.reset()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_P:
            self.filter_enabled = not self.filter_enabled
            self.filter_processor.reset()
            if self.stream:
                self._set_recording_status()
            else:
                filtre = "ON" if self.filter_enabled else "OFF"
                self.label_statut.setText(f"Filter: {filtre}")
                self.label_statut.setStyleSheet("color:#1976D2;")
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer_gui.stop()
        if self.stream:
            self.stream.stop()
            self.stream.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pg.setConfigOptions(useOpenGL=True, antialias=False, foreground="k", background="w")
    fenetre = EnregistreurHydrophoneWienerDualMic()
    fenetre.showFullScreen()
    sys.exit(app.exec_())