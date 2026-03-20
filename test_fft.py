import matplotlib.pyplot
import numpy as np
from scipy.io.wavfile import read

wav_path = "output/clean_record_sea.wav"
wav_file = read(wav_path)
print(wav_file)
wav_file = np.array(wav_file[1],dtype=float)

