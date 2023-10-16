import os
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

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
    layers.Dense(124416, activation='tanh'),
    layers.Reshape((288, 432, 1))
])

discriminator = tf.keras.Sequential([
    layers.Flatten(input_shape=(288, 432, 1)),
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


audio_files_dir = '/content/drive/MyDrive/audios/'
audio_files = os.listdir(audio_files_dir)

audio_data = []
for file in audio_files:
    file_path = os.path.join(audio_files_dir, file)
    data, sr = librosa.load(file_path, sr = 16_000) #downsample
    audio_data.append(data)

def preprocess_audio(audio_data):
    # Convert audio to mono
    #audio_data = librosa.to_mono(audio_data)

    # Resample audio to 16 kHz
    #audio_data = librosa.resample(audio_data, sr, 16000)

    # Normalize audio
    audio_data = librosa.util.normalize(audio_data)

    # Pad audio to fixed length of 1 second
    #audio_data = librosa.util.fix_length(audio_data, 16000)
    return audio_data

preprocessed_audio_data = []
for data in audio_data:
    preprocessed_data = preprocess_audio(data)
    preprocessed_audio_data.append(preprocessed_data)

tf_audio_data = tf.data.Dataset.from_tensor_slices(preprocessed_audio_data)
dataset = tf.data.Dataset.from_tensor_slices(preprocessed_audio_data)
print(f"DATASET SHAPE: {dataset.shape}")

# Preprocess and normalize dataset

# Training loop
epochs = 1000
batch_size = 20

for epoch in range(epochs):
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
        generator_loss = gan.train_on_batch(random_latent_vectors, misleading_labels)

    # Print progress
    print(f"Discriminator Loss: {discriminator_loss[0]} | Discriminator Accuracy: {discriminator_loss[1]}")
    print(f"Generator Loss: {generator_loss}")

    # Generate and save sample images
    if epoch % 10 == 0:
        print("Image saved")
        random_latent_vectors = tf.random.normal(shape=(2, latent_dim))
        generated_images = generator.predict(random_latent_vectors)

        """fig = plt.plot(generated_images)
        plt.savefig(f"/content/drive/MyDrive/data/generated_images_epoch_{epoch}.png")
        plt.close(fig)
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 4))
        for i, image in enumerate(generated_images):
            #print(i, image)
            axes[i].imshow(image.reshape((288, 432)), cmap='gray')
            axes[i].axis('off')
        plt.savefig(f"/content/drive/MyDrive/data/generated_images_epoch_{epoch}.png")
        plt.close(fig)
