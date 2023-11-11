import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import librosa
import os
from os import listdir
from os.path import isfile, join
import soundfile as sf

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    COLAB = True
    print("Note: using Google CoLab")
    %tensorflow_version 2.x
except:
    print("Note: not using Google CoLab")
    COLAB = False


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
    layers.Dense(1025*87, activation='tanh'),
    layers.Reshape((1025, 87))
])

discriminator = tf.keras.Sequential([
    #layers.Flatten(input_shape=(640, 480, 1)),
    layers.Flatten(input_shape=(1025, 87)),
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

# Load and preprocess dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/MyDrive/data/',
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



path = '/content/drive/MyDrive/audios/'
files = [f for f in listdir(path) if isfile(join(path, f))]
datasetSpectrums = []
sample_rate = 44100

def audio_spec_all(n):
    for i in range(n):
        try:
            xd = path + files[i]
            y, sr = librosa.load(xd)
            D = librosa.stft(y)  # STFT of y
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            #librosa.display.specshow(S_db, x_axis='time')
            #plt.xlabel(f"Archivo {files[i]}")
            #plt.savefig(f"specs/Test{i}.png")
            #print(i)
            #print(files[i])
            datasetSpectrums.append(S_db)
        except:
            print('Error')
    print('Fin')

audio_spec_all(len(files))
datasetSpectrums = tf.data.Dataset.from_tensor_slices(datasetSpectrums)
print("datasetSpectrum")
print(datasetSpectrums)

# Training loop
epochs = 1000
batch_size = 20

''' for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for real_images in dataset:
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
        generator_loss = gan.train_on_batch(random_latent_vectors, misleading_labels) '''

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for sound in datasetSpectrums:
        # Train discriminator
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        #print(f"Random Latent Vectors {random_latent_vectors.shape}")
        generated_images = generator.predict(random_latent_vectors)
        #print(f"Generated Images  {generated_images.shape}")
        combined_images = tf.concat([sound, generated_images], axis=0)
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

    # Generate and save sound
    if epoch % 2 == 0:
      random_latent_vectors = tf.random.normal(shape=(1, latent_dim))
      generated_sound = generator.predict(random_latent_vectors)
      abs_spectrogram = np.abs(librosa.core.spectrum.stft(generated_sound))
      audio_signal = librosa.core.spectrum.griffinlim(abs_spectrogram)
      print(audio_signal, audio_signal.shape)
      sf.write(f"/content/drive/MyDrive/data/generated_sound_epoch_{epoch}.wav", audio_signal, sample_rate, 'PCM_24')

'''     # Generate and save sample images
    if epoch % 2 == 0:
        print("Image saved")
        random_latent_vectors = tf.random.normal(shape=(1, latent_dim))
        generated_images = generator.predict(random_latent_vectors)
        plt.imshow(generated_images[0].reshape((640, 480)),cmap='gray');
        #plt.savefig(f"spec_generate/spec_{epoch}.png")
        plt.savefig(f"/content/drive/MyDrive/data/generated_images_epoch_{epoch}.png")
        plt.close() '''


'''     print("sound saved")
    random_latent_vectors = tf.random.normal(shape=(1, latent_dim))
        generated_images = generator.predict(random_latent_vectors)
        plt.imshow(generated_images[0].reshape((640, 480)),cmap='gray');
        #plt.savefig(f"spec_generate/spec_{epoch}.png")
        plt.savefig(f"/content/drive/MyDrive/data/generated_images_epoch_{epoch}.png")
        plt.close() '''
