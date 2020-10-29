import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np


# Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network
# CVPR 2016
class PixelShuffle(layers.Layer):
    def __init__(self, r, c_axis=3, name=None):
        super().__init__(name=name)
        self.c_axis = c_axis
        self.r = r

    def build(self, input_shape):
        assert input_shape[self.c_axis] % (self.r ** 2) == 0
        self.x = input_shape[1]
        self.y = input_shape[2]
        self.n_features = input_shape[self.c_axis] // (self.r ** 2)

    def _phase_shift(self, I):
       # Helper function with main phase shift operation
       X = tf.reshape(I, [-1, self.x, self.y, self.r, self.r])
       X = tf.split(X, self.x, 1)  # a, [bsize, b, r, r]
       X = tf.concat([tf.squeeze(x, 1) for x in X], 2)  # bsize, b, a*r, r
       X = tf.split(X, self.y, 1)  # b, [bsize, a*r, r]
       X = tf.concat([tf.squeeze(x, 1) for x in X], 2)  # bsize, a*r, b*r
       return tf.reshape(X, [-1, self.x * self.r, self.y * self.r, 1])

    def call(self, inputs):
        # inputs.shape[c_axis] should equal to n_features * r**2
        Xc = tf.split(inputs, self.n_features, self.c_axis)
        X = tf.concat([self._phase_shift(x) for x in Xc], self.c_axis)
        return X


# https://arxiv.org/pdf/1707.02937
def ICNRInitializer(shape, r):
    f = shape[-1]
    assert f % (r ** 2) == 0
    n = f // (r ** 2)

    class Init:
        def __init__(self, n, r):
            self.n = n
            self.r = r

        def __call__(self, shape, dtype='float32'):
            # last dim of shape must equal to n * r**2
            shape = tf.TensorShape(shape)
            sub_shape = shape.as_list()
            sub_shape[-1] = self.n
            z = tf.keras.initializers.glorot_normal()(sub_shape, dtype=dtype)
            z = tf.stack([z for _ in range(r ** 2)], axis=-1)
            z = tf.reshape(z, shape)
            return z

    return Init(n, r)


class PixelNormalization(layers.Layer):
    def __init__(self, c_axis=3, epsilon=1e-8, name=None):
        super().__init__(name=name)
        self.c_axis = c_axis
        self.epsilon = epsilon

    def call(self, inputs):
        return inputs / tf.sqrt(
            (tf.reduce_mean(tf.square(inputs), axis=self.c_axis, keepdims=True) + self.epsilon)
        )


class LayerNormalization(layers.Layer):
    # c_axis: axis of channel
    def __init__(self, c_axis, name=''):
        super().__init__(name=name)
        self.axis = c_axis
        self.eps = 1e-8

    def build(self, input_shape):
        num_channels = input_shape[self.axis]
        p_shape = [1 for _ in range(len(input_shape))]
        p_shape[self.axis] = num_channels
        self.beta = tf.Variable(tf.zeros_initializer()(shape=p_shape, dtype='float32'))
        self.gamma = tf.Variable(tf.ones_initializer()(shape=p_shape, dtype='float32'))

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True)
        norm_value = (inputs - mean) / tf.sqrt(var + self.eps)
        return norm_value * self.gamma + self.beta


class DenseSN(layers.Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=tf.keras.initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        # normalize it
        W_bar = W_reshaped / sigma
        # reshape weight tensor
        if not training:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class ConvSN2D(layers.Conv2D):

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=tf.keras.initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)

        # Set input spec.
        self.input_spec = tf.keras.layers.InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            # Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        # Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        # normalize it
        W_bar = W_reshaped / sigma
        # reshape weight tensor
        if not training:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)

        outputs = K.conv2d(
            inputs,
            W_bar,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class DownSampleBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, name='', pixnorm=False):
        super().__init__(name=name)
        self.conv = layers.Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer='glorot_normal')
        if pixnorm:
            self.pn = PixelNormalization()
        self.lrelu = layers.LeakyReLU(alpha=0.2)
        self.avgpool = layers.AveragePooling2D(pool_size=(2, 2))

    def call(self, inputs):
        x = inputs
        x = self.conv(x)
        if hasattr(self, 'pn'):
            x = self.pn(x)
        x = self.lrelu(x)
        x = self.avgpool(x)
        return x


class UpSampleBlock(tf.keras.Model):
    def __init__(self, in_features, out_features, kernel_size, name=''):
        super().__init__(name=name)
        self.conv = layers.Conv2D(out_features * 4,
                                  kernel_size,
                                  strides=1,
                                  padding='same',
                                  kernel_initializer=ICNRInitializer([kernel_size, kernel_size, in_features, out_features * 4], 2))
        self.pixelshuffle = PixelShuffle(2)

    def call(self, inputs):
        x = inputs
        x = self.conv(x)
        x = self.pixelshuffle(x)
        return x


def bicubicResize(img, orisize, tarsize, low, high):
    scale = tf.cast(tarsize, dtype='float32') / tf.cast(orisize, dtype='float32')
    return tf.clip_by_value(
        tf.raw_ops.ScaleAndTranslate(
            images=img,
            size=tarsize,
            scale=scale,
            translation=tf.zeros([2]),
            kernel_type='keyscubic',
            antialias=True), low, high
    )


def gaussianBlur(img, radius):
    sigma = radius / 2
    radius = tf.cast(K.round(radius), dtype='float32')
    x = tf.range(-radius, radius + 1, dtype='float32')
    y = tf.range(-radius, radius + 1, dtype='float32')
    x, y = tf.meshgrid(x, y)
    kernel = tf.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, 3])
    kernel = kernel[..., tf.newaxis]
    img = tf.pad(img, [[0, 0], [radius, radius], [radius, radius], [0, 0]], mode='SYMMETRIC')
    img = tf.nn.depthwise_conv2d(img, kernel, strides=[1, 1, 1, 1], padding='VALID')
    return img


def getRandomRadius():
    radius = tf.math.floor(tf.random.uniform([], 0, 1) * 6)
    return radius


def blurDown(img, orisize, tarsize, radius):
    if radius > 0:
        img = gaussianBlur(img, radius)
    down = bicubicResize(img, orisize, tarsize, -1., 1.)
    return down


def psnr(img1, img2):
    img1 = tf.cast(img1, tf.float32)
    img2 = tf.cast(img2, tf.float32)
    return 10 * tf.math.log(
        (255 ** 2) / (tf.reduce_mean(tf.square(img1 - img2), axis=[1, 2, 3]))
    ) / tf.math.log(10.)


def ssim(img1, img2):
    img1 = tf.cast(img1, tf.float32)
    img2 = tf.cast(img2, tf.float32)
    n = img1.shape[1] * img1.shape[2] * img1.shape[3]
    c1 = 6.5025
    c2 = 58.5225
    c3 = 29.26125
    mu1 = tf.reduce_mean(img1, axis=[1, 2, 3], keepdims=True)
    mu2 = tf.reduce_mean(img2, axis=[1, 2, 3], keepdims=True)
    sigma1 = tf.sqrt(tf.reduce_sum(tf.square(img1 - mu1), axis=[1, 2, 3], keepdims=True) / (n - 1))
    sigma2 = tf.sqrt(tf.reduce_sum(tf.square(img2 - mu2), axis=[1, 2, 3], keepdims=True) / (n - 1))
    sigma12 = tf.reduce_sum(tf.math.multiply(img1 - mu1, img2 - mu2), axis=[1, 2, 3], keepdims=True) / (n - 1)
    l = (2 * mu1 * mu2 + c1) / (mu1 ** 2 + mu2 ** 2 + c1)
    c = (2 * sigma1 * sigma2 + c2) / (sigma1 ** 2 + sigma2 ** 2 + c2)
    s = (sigma12 + c3) / (sigma1 * sigma2 + c3)
    return l * c * s


def meanVar(img):
    h_kernel = tf.constant([[-1, 1]], dtype='float32')
    h_kernel = tf.tile(h_kernel[..., tf.newaxis], [1, 1, 3])
    h_kernel = h_kernel[..., tf.newaxis]
    var_h = tf.nn.depthwise_conv2d(img, h_kernel, strides=[1, 1, 1, 1], padding='VALID')[:, :-1, :, :]
    v_kernel = tf.constant([[-1], [1]], dtype='float32')
    v_kernel = tf.tile(v_kernel[..., tf.newaxis], [1, 1, 3])
    v_kernel = v_kernel[..., tf.newaxis]
    var_v = tf.nn.depthwise_conv2d(img, v_kernel, strides=[1, 1, 1, 1], padding='VALID')[:, :, :-1, :]
    return tf.reduce_mean(tf.math.abs(var_h) + tf.math.abs(var_v))
