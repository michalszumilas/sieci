import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

sprites = np.load(r'C:\Users\abeod\Desktop\sieci\sprites.npy')

sprites_labels = np.load(r'C:\Users\abeod\Desktop\sieci/sprites_labels.npy')

# Check their shapes to understand the structure
print(sprites.shape)  # Shape of the sprite images data
print(sprites_labels.shape)  # Shape of the corresponding labels


# Load the labels CSV
labels_df = pd.read_csv(r'C:\Users\abeod\Desktop\sieci/labels.csv')

# Inspect the first few rows of the DataFrame
print(labels_df.head())
from skimage.transform import resize

# Assuming sprites are in a 3D array (num_images, height, width)
# Resize images to a fixed size, e.g., 32x32
resized_sprites = np.array([resize(sprite, (32, 32)) for sprite in sprites])

# Normalize pixel values to range [0, 1]
normalized_sprites = resized_sprites / 255.0
# Assuming labels in the CSV are in the same order as the sprites
# Merge the image data with the labels
image_label_pairs = list(zip(normalized_sprites, labels_df['description']))  # Adjust column name as necessary

# Example: Display the first image-description pair
import matplotlib.pyplot as plt

plt.imshow(image_label_pairs[0][0])  # Display the first sprite image
print(image_label_pairs[0][1])  # Print the corresponding description

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def build_generator(latent_dim, num_classes):
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim + num_classes))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(32 * 32 * 3, activation='tanh'))
    model.add(layers.Reshape((32, 32, 3)))
    return model


def build_discriminator(img_shape):
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=img_shape))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


# Funkcja do treningu GAN
def train_gan(generator, discriminator, gan, dataset, labels_one_hot, latent_dim, epochs=10000, batch_size=64):
    real = np.ones((batch_size, 1))  # Real label
    fake = np.zeros((batch_size, 1))  # Fake label

    for epoch in range(epochs):
        # Trening dyskryminatora
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        real_imgs = dataset[idx]
        label_idx = np.random.randint(0, num_classes, batch_size)
        label_vec = labels_one_hot[label_idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        noise_with_labels = np.concatenate([noise, label_vec], axis=1)

        # Trening na rzeczywistych i fałszywych obrazach
        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        gen_imgs = generator.predict(noise_with_labels)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Trening generatora
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        noise_with_labels = np.concatenate([noise, label_vec], axis=1)
        g_loss = gan.train_on_batch(noise_with_labels, real)

        # Drukowanie postępu
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")


# Parametry
latent_dim = 100
img_shape = (32, 32, 3)

# Budowa i kompilacja modelu
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

generator = build_generator(latent_dim, num_classes)
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Trening GAN
train_gan(generator, discriminator, gan, normalized_images, labels_one_hot, latent_dim, epochs=10000, batch_size=64)
