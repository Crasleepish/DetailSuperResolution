import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from CustomUtils import *


class ResidualBlock(tf.keras.Model):
    def __init__(self, input_features, output_features, name=None):
        super().__init__(name=name)
        self.input_features = input_features
        self.output_features = output_features
        self.lrelu_1 = layers.LeakyReLU(alpha=0.2)
        self.conv2d_1 = ConvSN2D(filters=self.input_features,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      use_bias=True,
                                      kernel_initializer='glorot_normal')
        self.lrelu_2 = layers.LeakyReLU(alpha=0.2)
        self.avgpool_1 = layers.AveragePooling2D(pool_size=(2, 2))
        self.conv2d_2 = ConvSN2D(filters=self.output_features,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding='same',
                          use_bias=True,
                          kernel_initializer='glorot_normal')
        self.avgpool_2 = layers.AveragePooling2D(pool_size=(2, 2))
        self.conv2d_3 = ConvSN2D(filters=self.output_features,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 padding='valid',
                                 use_bias=True,
                                 kernel_initializer='glorot_normal')

    def call(self, inputs):
        shortcut = inputs

        x = self.lrelu_1(inputs)

        x = self.conv2d_1(x)
        x = self.lrelu_2(x)
        # (w, h, input_features)

        x = self.avgpool_1(x)
        x = self.conv2d_2(x)
        # (w/2, h/2, output_features)

        shortcut = self.avgpool_2(shortcut)
        shortcut = self.conv2d_3(shortcut)
        # (w/2, h/2, output_features)

        x = layers.add([x, shortcut])
        return x


class FeatureInputTail(tf.keras.Model):
    def __init__(self, output_features, name=None):
        super().__init__(name=name)
        self.output_features = output_features
        self.conv2d = ConvSN2D(filters=self.output_features,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same',
                                    use_bias=True,
                                    kernel_initializer='glorot_normal')
        self.lrelu = layers.LeakyReLU(alpha=0.2)

        self.conv2d_2 = ConvSN2D(filters=self.output_features,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same',
                                    use_bias=True,
                                    kernel_initializer='glorot_normal')

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.lrelu(x)
        x = self.conv2d_2(x)
        return x


class fadeLayer(layers.Layer):
    def __init__(self, max_count, name=None):
        super().__init__(name=name)
        self.max_count = max_count
        self.current_count = tf.Variable(initial_value=0., trainable=False, dtype='float32')

    def call(self, inputs):
        low_x = inputs[0]
        high_x = inputs[1]
        alpha = K.clip(self.current_count / self.max_count, 0., 1.)
        return high_x * alpha + low_x * (1 - alpha)

    def update_count(self):
        self.current_count.assign_add(1)


# ConvSN&DenseSN: https://arxiv.org/pdf/1802.05957
class ModelFactory:
    def __init__(self):
        self.float_layer_count = 0
        self.complete_d = self.get_discrimnate_model_complete()

    def get_discrimnate_model_complete(self):
        input = tf.keras.Input(shape=(128, 128, 12), dtype=tf.float32)

        x = ConvSN2D(filters=32,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding='same',
                          use_bias=True,
                          kernel_initializer='glorot_normal')(input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = ConvSN2D(filters=32,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding='same',
                          use_bias=True,
                          kernel_initializer='glorot_normal')(x)
        # output (128, 128, 32)

        x = ResidualBlock(32, 64)(x)
        # output (64, 64, 64)
        x = ResidualBlock(64, 128)(x)
        # output (32, 32, 128)
        x = ResidualBlock(128, 256)(x)
        # output (16, 16, 256)
        x = ResidualBlock(256, 512)(x)
        # output (8, 8, 512)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Flatten()(x)
        x = DenseSN(units=1, activation=None, kernel_initializer='glorot_normal')(x)

        model = tf.keras.models.Model(inputs=input, outputs=x)
        return model

    def get_d_model(self):
        return self.complete_d
