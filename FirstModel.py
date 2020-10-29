from tensorflow.keras import layers
from CustomUtils import *


class FirstModelResidualBlock(tf.keras.Model):
    def __init__(self, features, name=None):
        super().__init__(name=name)
        self.features = features
        self.prelu_1 = layers.PReLU(shared_axes=[1, 2])
        self.conv2d_1 = layers.Conv2D(filters=self.features,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      use_bias=True,
                                      kernel_initializer='glorot_normal')
        self.prelu_2 = layers.PReLU(shared_axes=[1, 2])
        self.conv2d_2 = layers.Conv2D(filters=self.features,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      use_bias=True,
                                      kernel_initializer='glorot_normal')
        self.conv2d_3 = layers.Conv2D(filters=self.features,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding='valid',
                                      use_bias=True,
                                      kernel_initializer='glorot_normal')

    def call(self, inputs):
        shortcut = inputs

        x = self.prelu_1(inputs)

        x = self.conv2d_1(x)
        x = self.prelu_2(x)

        x = self.conv2d_2(x)

        shortcut = self.conv2d_3(shortcut)

        x = layers.add([x, shortcut])
        return x


def get_first_model():
    input = tf.keras.Input(shape=(64, 64, 3), dtype=tf.float32)
    x = layers.Conv2D(filters=64,
                      kernel_size=(9, 9),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='glorot_normal')(input)  # (64,64,64)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    x = layers.Conv2D(filters=64,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='glorot_normal')(x)
    # (64,64,64)

    shortcut = x

    x = FirstModelResidualBlock(64)(x)
    x = FirstModelResidualBlock(64)(x)
    x = FirstModelResidualBlock(64)(x)
    x = FirstModelResidualBlock(64)(x)
    x = FirstModelResidualBlock(64)(x)  # (64,64,64)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    x = layers.Conv2D(filters=64,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='glorot_normal')(x)
    shortcut = layers.Conv2D(filters=64,
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             padding='valid',
                             use_bias=True,
                             kernel_initializer='glorot_normal')(shortcut)
    x = layers.add([x, shortcut])
    x = layers.PReLU(shared_axes=[1, 2])(x)

    x = UpSampleBlock(64, 64, 3, name='up_1')(x)  # (128, 128, 64)

    x = FirstModelResidualBlock(64)(x)  # (128, 128, 64)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    x = UpSampleBlock(64, 64, 3, name='up_2')(x)  # (256, 256, 64)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    x = layers.Conv2D(filters=3,
                      kernel_size=(9, 9),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='glorot_normal')(x)   # (256, 256, 3)
    x = layers.Activation('tanh')(x)

    model = tf.keras.models.Model(inputs=input, outputs=x)
    return model
