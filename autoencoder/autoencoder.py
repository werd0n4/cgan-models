from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, add, BatchNormalization , Dropout, LeakyReLU
from tensorflow.keras import Model
import keras.datasets.cifar10 as cf10
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.python.util.compat import as_str

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = cf10.load_data()
    noise_factor = 0.11
    
    x_train = x_train / 255
    x_test = x_test / 255

    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    
    input = Input(shape=(32, 32, 3))

    # Encoder
    conv_layer_1 = Conv2D(32, 3, activation='relu', padding='same')(input)
    x = BatchNormalization()(conv_layer_1)
    x = MaxPooling2D()(x)
    #x = Dropout(0.05)(x)

    conv_layer_2 = Conv2D(32, 3, padding='same')(x) 
    x = LeakyReLU()(conv_layer_2)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    #x = Dropout(0.05)(x)

    conv_layer_3 = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(conv_layer_3)
    encoded = MaxPooling2D()(x)


    # Decoder
    x = Conv2DTranspose(64, 3,activation='relu',strides=(2,2), padding='same')(encoded)
    x = add([x, conv_layer_3])
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)

    x = Conv2DTranspose(32, 3, activation='relu',strides=(2,2), padding='same')(x)
    x = add([x,conv_layer_2]) 
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)

    x = Conv2DTranspose(32, 3, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    decoded = Conv2DTranspose(3, 3, activation='sigmoid',strides=(2,2), padding='same')(x)


    # Autoencoder
    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.summary()

    #autoencoder.fit(x_train_noisy, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test_noisy, x_test))
    #autoencoder.save_weights('MGU\\cgan-models\\autoencoder\\saved-model\\model', overwrite=True)
    autoencoder.load_weights('MGU\\cgan-models\\autoencoder\\saved-model\\model')

    predicted = autoencoder.predict(x_test_noisy)

    columns = 6
    rows = 3
    size = 1.6
    ds_offset = 10

    figure = plt.figure(figsize=(size*columns, size*rows))
    figure.tight_layout()

    for i in range(columns):
        figure.add_subplot(rows, columns, i + 1)
        plt.imshow(x_test[ds_offset + i])
        plt.axis("off")

    for i in range(columns):
        figure.add_subplot(rows, columns, columns + i + 1)
        plt.imshow(x_test_noisy[ds_offset + i])
        plt.axis("off")

    for i in range(columns):
        figure.add_subplot(rows, columns, 2*columns + i + 1)
        plt.imshow(predicted[ds_offset + i])
        plt.axis("off")

    plt.savefig('MGU\\cgan-models\\autoencoder\\results\\result.png')

    plt.show()

