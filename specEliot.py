
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# read file
file = '/Users/elliot_hernandez/Documents/Elliot/UNAM/Tutoría/Sonidos/otros/SC_230614_164449.aiff'
y, sr = librosa.load(file, sr=44100)
librosa.display.waveshow(y, sr=sr)

# process
abs_spectrogram = np.abs(librosa.core.spectrum.stft(y))
audio_signal = librosa.core.spectrum.griffinlim(abs_spectrogram)

print(audio_signal, audio_signal.shape)


import soundfile as sf
sample_rate = 44100

sf.write('test.wav', audio_signal, sample_rate, 'PCM_24')