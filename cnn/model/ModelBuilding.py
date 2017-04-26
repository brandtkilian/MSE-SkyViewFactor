from __future__ import absolute_import
from __future__ import print_function

from keras.layers.noise import GaussianNoise
import keras.models as models
from keras.layers.core import Dropout, Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers


def create_encoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
        #Dropout(0.5),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
        #Dropout(0.25),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
        #Dropout(0.25),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        #MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]


def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        #UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size, pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size, pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size, pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
    ]


def create_model(width, height, nblbl):
    data_shape = width * height
    autoencoder = models.Sequential()
    autoencoder.add(ZeroPadding2D(input_shape=(3, height, width)))
    autoencoder.add(GaussianNoise(sigma=0.3))
    autoencoder.encoding_layers = create_encoding_layers()
    autoencoder.decoding_layers = create_decoding_layers()
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    autoencoder.add(Convolution2D(nblbl, 1, 1, border_mode='valid', ))
    autoencoder.add(Reshape((nblbl, data_shape), input_shape=(nblbl, height, width)))
    autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('softmax'))
    adad = optimizers.Adadelta()
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


    autoencoder.compile(loss="categorical_crossentropy", optimizer=adad, metrics=['accuracy'])
    return autoencoder
