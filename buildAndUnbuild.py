import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

#print(librosa.show_versions())
file = 'audios/SC_230614_164449.wav'
y, sr = librosa.load(file, sr=44100, mono=True)
print(y)
spectrum = np.abs(librosa.stft(y))
print(spectrum)

fig, ax = plt.subplots()
#img = librosa.display.specshow(librosa.amplitude_to_db(spectrum,
#                                                       ref=np.max),
#                               y_axis='log', x_axis='time', ax=ax)
#ax.set_title('Power spectrogram')
#fig.colorbar(img, ax=ax, format="%+2.0f dB")

img = librosa.display.specshow(spectrum)
plt.show()
