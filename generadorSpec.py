import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa
from os import listdir
from os.path import isfile, join

path = 'audios/'
files = [f for f in listdir(path) if isfile(join(path, f))]

#y, sr = librosa.load('/Users/elliot_hernandez/Documents/Elliot/UNAM/Tutoría/Sonidos/Otros/SC_230614_164449.aiff')
#D = librosa.stft(y)  # STFT of y
#S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

#plt.figure()
#librosa.display.specshow(S_db)
#plt.colorbar()

def audio_spec_all(n):
    for i in range(n):
        try:
            xd = path + files[i]
            y, sr = librosa.load(xd)
            D = librosa.stft(y)  # STFT of y
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            librosa.display.specshow(S_db, x_axis='time')
            #plt.xlabel(f"Archivo {files[i]}")
            plt.savefig(f"specs/Test{i}.png")
            print(i)
            print(files[i])
        except:
            print('Error')
    print('Fin')

audio_spec_all(len(files))