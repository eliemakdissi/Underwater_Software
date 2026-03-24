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
from utils.realtime_export import (
    create_run_folder,
    export_spectrogram_png,
    export_waveform_png,
)
from utils.realtime_wiener import BandpassWienerProcessor


TARGET_FREQ_HZ = 8000.0
TARGET_BANDWIDTH_HZ = 1000.0
FILTER_ORDER = 4
WIENER_WINDOW = 31


class EnregistreurHydrophoneWiener(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hydrophone Recorder (Wiener Test)")
        self.samplerate = 192000
        self.audio_data_for_save = []
        self.audio_queue = queue.Queue()
        self.stream = None
        self.choix_canal = 1

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

        self.session_waveform_full = []
        self.fft_size = 8192
        self.hop_size = 4096
        self.duree_visible = 5
        self.overlap_buffer = np.array([], dtype=np.float32)

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

        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                self.combo_in.addItem(f"{i}: {d['name'][:20]}", i)
            if d["max_output_channels"] > 0:
                self.combo_out.addItem(f"{i}: {d['name'][:20]}", i)

        try:
            idx_in = self.combo_in.findData(sd.default.device[0])
            idx_out = self.combo_out.findData(sd.default.device[1])
            if idx_in >= 0:
                self.combo_in.setCurrentIndex(idx_in)
            if idx_out >= 0:
                self.combo_out.setCurrentIndex(idx_out)
        except Exception:
            pass

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
        self.combo_canaux.addItems(["1 (Mono)", "2 (Stereo)"])

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

        layout_principal.addWidget(lbl_info_filtre)
        layout_principal.addLayout(layout_top)

        self.hist_fft_full = None
        self.img_width = self._recalculer_img_width()
        self.hist_onde_full = np.zeros(self.duree_visible * self.samplerate, dtype=np.float32)
        self.t_axis_onde = np.linspace(0, self.duree_visible, len(self.hist_onde_full), endpoint=False)

        self.win_graph = pg.GraphicsLayoutWidget()
        layout_principal.addWidget(self.win_graph)

        self.plot_onde = self.win_graph.addPlot(title="Waveform")
        self.plot_onde.setLabel("left", "Amplitude")
        self.plot_onde.setLabel("bottom", "Time", units="s")
        self.plot_onde.setYRange(-1.0, 1.0)
        self.plot_onde.setXRange(0, self.duree_visible, padding=0)
        self.plot_onde.showGrid(x=True, y=True, alpha=0.3)
        self.curve_onde = self.plot_onde.plot(
            pen=pg.mkPen("b", width=2), autoDownsample=True, clipToView=True
        )

        self.win_graph.nextRow()

        self.plot_spectro = self.win_graph.addPlot(title="Spectrogram")
        self.plot_spectro.setLabel("left", "Frequency", units="kHz")
        self.plot_spectro.setLabel("bottom", "Time", units="s")
        self.plot_spectro.setYRange(0, self.samplerate / 2000, padding=0)
        self.plot_spectro.setXRange(0, self.duree_visible, padding=0)

        self.img_spectro = pg.ImageItem()
        self.plot_spectro.addItem(self.img_spectro)

        cmap = pg.colormap.get("inferno")
        bar = pg.ColorBarItem(values=(-100, -20), colorMap=cmap, label="Power (dB)")
        bar.setImageItem(self.img_spectro)

        layout_principal.setStretchFactor(self.win_graph, 1)
        widget_central.setLayout(layout_principal)
        self.setCentralWidget(widget_central)

        self.timer_gui = QTimer()
        self.timer_gui.timeout.connect(self.update_charts)

    def _get_hop_ratio(self):
        return {"1/1": 1.0, "1/2": 0.5, "1/4": 0.25, "1/8": 0.125}.get(
            self.combo_hop.currentText(), 0.5
        )

    def _recalculer_img_width(self):
        return max(1, int((self.duree_visible * self.samplerate) / self.hop_size))

    def _reset_spectro(self):
        self.hist_fft_full = None
        self.overlap_buffer = np.array([], dtype=np.float32)
        if hasattr(self, "img_spectro"):
            self.img_spectro.clear()

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
        self.hist_onde_full = np.zeros(self.duree_visible * self.samplerate, dtype=np.float32)
        self.t_axis_onde = np.linspace(0, self.duree_visible, len(self.hist_onde_full), endpoint=False)
        self.img_width = self._recalculer_img_width()
        self._reset_spectro()
        if hasattr(self, "plot_onde"):
            self.plot_onde.setXRange(0, self.duree_visible, padding=0)
        if hasattr(self, "plot_spectro"):
            self.plot_spectro.setXRange(0, self.duree_visible, padding=0)

    def changer_parametres_calcul(self):
        self.fft_size = int(self.combo_fft.currentText())
        self.hop_size = int(self.fft_size * self._get_hop_ratio())
        self.img_width = self._recalculer_img_width()
        self._reset_spectro()

    def callback_audio(self, indata, outdata, frames, time, status):
        data_in = indata.astype(np.float32, copy=True)
        data_proc = self._apply_optional_filter(data_in)

        self.audio_data_for_save.append(data_proc.copy())
        if self.stream:
            mono = data_proc[:, 0].copy()
            self.audio_queue.put(mono)
            self.session_waveform_full.append(mono)

        if data_proc.shape[1] == 1 and outdata.shape[1] == 2:
            outdata[:, 0] = data_proc[:, 0]
            outdata[:, 1] = data_proc[:, 0]
        elif data_proc.shape[1] == outdata.shape[1]:
            outdata[:] = data_proc
        else:
            for ch in range(outdata.shape[1]):
                outdata[:, ch] = data_proc[:, 0]

    def update_charts(self):
        all_new_data = []
        while not self.audio_queue.empty():
            all_new_data.append(self.audio_queue.get())
        if not all_new_data:
            return

        raw_audio = np.concatenate(all_new_data)
        crop = raw_audio[-len(self.hist_onde_full) :] if len(raw_audio) > len(self.hist_onde_full) else raw_audio
        self.hist_onde_full = np.roll(self.hist_onde_full, -len(crop))
        self.hist_onde_full[-len(crop) :] = crop
        skip = max(1, len(self.hist_onde_full) // 4000)
        self.curve_onde.setData(x=self.t_axis_onde[::skip], y=self.hist_onde_full[::skip])

        audio_to_process = np.concatenate([self.overlap_buffer, raw_audio])
        window = np.hanning(self.fft_size)
        new_fft_cols = []
        idx = 0
        while idx + self.fft_size <= len(audio_to_process):
            chunk = audio_to_process[idx : idx + self.fft_size]
            fft_complex = np.fft.rfft(chunk * window) / self.fft_size
            fft_db = 20 * np.log10(np.abs(fft_complex) + 1e-9)
            new_fft_cols.append(fft_db)
            idx += self.hop_size
        self.overlap_buffer = audio_to_process[idx:]

        if not new_fft_cols:
            return

        new_data_matrix = np.column_stack(new_fft_cols)
        n_cols = new_data_matrix.shape[1]
        num_freq_bins = new_data_matrix.shape[0]

        if (
            self.hist_fft_full is None
            or self.hist_fft_full.shape[0] != num_freq_bins
            or self.hist_fft_full.shape[1] != self.img_width
        ):
            self.hist_fft_full = np.full((num_freq_bins, self.img_width), -100, dtype=np.float32)

        self.hist_fft_full = np.roll(self.hist_fft_full, -n_cols, axis=1)
        self.hist_fft_full[:, -n_cols:] = new_data_matrix
        self.img_spectro.setImage(self.hist_fft_full.T, autoLevels=False, levels=[-100, -20])
        self.img_spectro.setRect(QRectF(0, 0, self.duree_visible, self.samplerate / 2000))

    def demarrer_enregistrement(self):
        self.audio_data_for_save = []
        self.session_waveform_full = []
        self.choix_canal = self.combo_canaux.currentIndex() + 1
        self.overlap_buffer = np.array([], dtype=np.float32)
        self.hist_fft_full = None
        self.filter_processor.reset()
        self.hist_onde_full.fill(0)

        self.dossier_session = create_run_folder(self.output_root, prefix="run")

        self.fft_size = int(self.combo_fft.currentText())
        self.hop_size = int(self.fft_size * self._get_hop_ratio())
        self.img_width = self._recalculer_img_width()

        idx_in = self.combo_in.currentData()
        idx_out = self.combo_out.currentData()

        try:
            self.stream = sd.Stream(
                device=(idx_in, idx_out),
                samplerate=self.samplerate,
                channels=(self.choix_canal, 2),
                blocksize=self.fft_size,
                callback=self.callback_audio,
                dtype="float32",
            )
            self.stream.start()
            self.timer_gui.start(50)

            self._set_recording_status()
            self.btn_demarrer.setEnabled(False)
            self.btn_arreter.setEnabled(True)
            self.combo_canaux.setEnabled(False)
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
            nom_fichier = f"hydro_{horodatage}.wav"
            dossier_cible = self.dossier_session or os.path.join(self.output_root, f"run_{horodatage}")
            os.makedirs(dossier_cible, exist_ok=True)

            wavfile.write(
                os.path.join(dossier_cible, nom_fichier),
                self.samplerate,
                enregistrement_complet,
            )

            audio_mono = (
                np.concatenate(self.session_waveform_full)
                if self.session_waveform_full
                else enregistrement_complet[:, 0]
            )
            try:
                export_waveform_png(audio_mono, self.samplerate, dossier_cible)
                export_spectrogram_png(
                    audio_mono,
                    self.samplerate,
                    self.fft_size,
                    self.hop_size,
                    dossier_cible,
                )
            except Exception as export_error:
                print(f"Erreur export figures: {export_error}")

            self.label_statut.setText(f"OK {nom_fichier} ({os.path.basename(dossier_cible)})")
            self.label_statut.setStyleSheet("color:#388E3C;")
        else:
            self.label_statut.setText("")

        self.btn_demarrer.setEnabled(True)
        self.btn_arreter.setEnabled(False)
        self.combo_canaux.setEnabled(True)
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
    pg.setConfigOptions(antialias=True, foreground="k", background="w")
    fenetre = EnregistreurHydrophoneWiener()
    fenetre.showFullScreen()
    sys.exit(app.exec_())
