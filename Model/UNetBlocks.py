import tensorflow as tf
from tensorflow.keras import layers

class DownscaleBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=15, padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv1D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.1)
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.pool = layers.MaxPool1D(2,2)

    def call(self, input_tensor):
        x = self.convA(input_tensor)
        x = self.bn2a(x)
        x = self.reluA(x)

        p = self.pool(x)
        
        return x, p


class UpscaleBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=5, padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.us = layers.UpSampling1D(2)
        self.convA = layers.Conv1D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.1)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.conc = layers.Concatenate()

    def call(self, x, skip):
        x = self.us(x)
        concat = self.conc([x, skip])
        x = self.convA(concat)
        x = self.bn2a(x)
        x = self.reluA(x)

        return x


class BottleNeckBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=15, padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv1D(filters, kernel_size, strides, padding)
        self.bnA = tf.keras.layers.BatchNormalization()
        self.reluA = layers.LeakyReLU(alpha=0.1)

    def call(self, x):
        x = self.convA(x)
        x = self.bnA(x)
        x = self.reluA(x)
        
        return x