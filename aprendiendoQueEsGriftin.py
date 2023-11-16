import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import librosa
import os
from os import listdir
from os.path import isfile, join

#constantes
frame_length =255
frame_step=128

# Load and preprocess dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'specs/',
    #image_size=(28, 28),
    label_mode=None,
    image_size=(640, 480),
    batch_size=20,
    shuffle=True,
    color_mode='rgb'
)



datasetAudio = tf.keras.utils.audio_dataset_from_directory(
    'audios/',
    #image_size=(28, 28),
    labels=None,
    label_mode=None,
    #sampling_rate=44100,
    batch_size=20,
    shuffle=True,
    output_sequence_length=88200
)
print("datasetAudio")
print(datasetAudio)


def squeeze(audio):
  audio = tf.squeeze(audio, axis=-1)
  return audio

print("datasetAudio")
print(datasetAudio)

datasetAudio = datasetAudio.map(squeeze, tf.data.AUTOTUNE)


def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=frame_length, frame_step=frame_step)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio: (get_spectrogram(audio)),
      num_parallel_calls=tf.data.AUTOTUNE)

datasetSpectrums = make_spec_ds(datasetAudio)
print("spectrums")
print(datasetSpectrums)

generated_spectrum = datasetSpectrums.take(0)
print(generated_spectrum)
#abs_spectrogram = tf.abs(generated_spectrum)
audio_signal = librosa.griffinlim(generated_spectrum)
#print(audio_signal, audio_signal.shape)
#inverse_stft = tf.signal.inverse_stft(
#    generated_spectrum, frame_length, frame_step,
#    window_fn=tf.signal.inverse_stft_window_fn(frame_step))


