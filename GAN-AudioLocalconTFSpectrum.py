import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import librosa
import os
from os import listdir
from os.path import isfile, join

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

# Preprocess and normalize dataset
#dataset = dataset.map(lambda x, _: (x / 255.0) * 2 - 1)
dataset = dataset.map(lambda x: (tf.image.rgb_to_grayscale(x) / 255.0) * 2 - 1)
print("dataset")
print(dataset)

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
      waveform, frame_length=255, frame_step=128)
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
# Training loop
epochs = 1000
batch_size = 20

for example_spectrograms in datasetSpectrums.take(1):
  break

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)




####EJEMPLO
# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=datasetSpectrums.map(map_func=lambda spec: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    #layers.Dense(num_labels),
])

model.summary()
####TERMINA EJEMPLO

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define GAN architecture
latent_dim = 100

generator = tf.keras.Sequential([
    layers.Dense(256, input_shape=(latent_dim,), activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    #layers.Dense(784, activation='tanh'),
    layers.Dense(688*129, activation='tanh'),
    layers.Reshape((688,129,1))
])

generator.summary()

discriminator = tf.keras.Sequential([
    layers.Flatten(input_shape=(688,129,1)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

discriminator.summary()

# Compile discriminator
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])

# Compile GAN
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss=tf.keras.losses.BinaryCrossentropy())

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for real_images in datasetSpectrums:
        # Train discriminator
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        #print(f"Random Latent Vectors {random_latent_vectors.shape}")
        generated_images = generator.predict(random_latent_vectors)
        #print(f"Generated Images  {generated_images.shape}")
        combined_images = tf.concat([real_images, generated_images], axis=0)
        #combined_images = tf.stack(combined_images)
        #print(f"Combined Images {combined_images.shape}")
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        #labels = tf.stack(labels)
        #print(f"Labels {labels.shape}")
        discriminator_loss = discriminator.train_on_batch(combined_images, labels)

        # Train generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        misleading_labels = tf.ones((batch_size, 1))
        generator_loss = gan.train_on_batch(random_latent_vectors, misleading_labels)

    # Print progress
    print(f"Discriminator Loss: {discriminator_loss[0]} | Discriminator Accuracy: {discriminator_loss[1]}")
    print(f"Generator Loss: {generator_loss}")

    # Generate and save sample images
    if epoch % 2 == 0:
        print("Spectrograma generado nos ganamos una chelita saved")
        random_latent_vectors = tf.random.normal(shape=(1, latent_dim))
        generated_images = generator.predict(random_latent_vectors)
        print(generated_images)
        #plt.imshow(generated_images[0].reshape((640, 480)),cmap='gray');
        #plt.savefig(f"spec_generate/spec_{epoch}.png")
        #plt.close()
