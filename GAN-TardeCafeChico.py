import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import librosa
import os


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
    layers.Dense(88200, activation='tanh'),
    #layers.Reshape((288, 432, 1))
])

discriminator = tf.keras.Sequential([
    #layers.Flatten(input_shape=(288, 432, 1)),
    layers.Flatten(input_shape=(88200,1)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

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


audio_files_dir = 'audios/'
audio_files = os.listdir(audio_files_dir)

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  print(audio.shape)
  return tf.squeeze(audio, axis=-1)

audio_data = []
for file in audio_files:
    test_file = tf.io.read_file(audio_files_dir + file)
    waveform = decode_audio(test_file)
    audio_data.append(waveform)

files_ds = tf.data.Dataset.from_tensor_slices(audio_data)
print(files_ds)


# Training loop
epochs = 1000
batch_size = 20

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for audio in files_ds:
        # Train discriminator
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        #print(f"Random Latent Vectors {random_latent_vectors.shape}")
        generated_images = generator.predict(random_latent_vectors)
        #print(f"Generated Images  {generated_images.shape}")
        combined_images = tf.concat([audio, generated_images], axis=0)
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

