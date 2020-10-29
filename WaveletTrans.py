import tensorflow as tf

# up scale by tensorflow
def UpSample(x, axis=1):
    shape = x.shape.as_list()
    z = tf.zeros_like(x)
    z = tf.stack([x, z], axis=axis+1)
    shape[axis] = shape[axis]*2
    z = tf.reshape(z, shape)
    loc = [slice(0, None) for i in shape]
    loc[axis] = slice(None, -1)
    return z[loc]


def DownSample(x, axis=1):
    shape = x.shape.as_list()
    loc = [slice(0, None) for i in shape]
    loc[axis] = slice(1, None, 2)
    return x[loc]


def UpScale(img, detail):
    #detail: D,H,V
    h_psy = tf.constant([[-1],
                         [1]], dtype='float32')
    h_psy = tf.expand_dims(tf.stack([h_psy, h_psy, h_psy], axis=2), axis=3)  # (2, 1, 3, 1)
    h_phi = tf.constant([[1],
                         [1]], dtype='float32')
    h_phi = tf.expand_dims(tf.stack([h_phi, h_phi, h_phi], axis=2), axis=3)  # (2, 1, 3, 1)
    kernel = tf.concat([h_psy, h_psy, h_phi, h_phi], axis=2)  # (2, 1, 12, 1)
    data = tf.concat([detail, img], axis=3)  # (B, H, W, 12)
    data = UpSample(data, axis=1)  # (B, 2H-1, W, 12)
    data = tf.pad(data, [[0, 0], [1, 1], [0, 0], [0, 0]])
    p = tf.expand_dims(tf.expand_dims(tf.concat([tf.eye(6), tf.eye(6)], axis=0), axis=0), axis=0)  # (1, 1, 12, 6)
    data = tf.nn.separable_conv2d(data, kernel, p, strides=[1, 1, 1, 1], padding='VALID')
    # (B, 2H, W, 6)
    data = UpSample(data, axis=2)  # (B, 2H, 2W-1, 6)
    h_psy = tf.transpose(h_psy, perm=[1, 0, 2, 3])
    h_phi = tf.transpose(h_phi, perm=[1, 0, 2, 3])
    kernel = tf.concat([h_psy, h_phi], axis=2)  # (1, 2, 6, 1)
    p = tf.expand_dims(tf.expand_dims(tf.concat([tf.eye(3), tf.eye(3)], axis=0), axis=0), axis=0)  # (1, 1, 6, 3)
    data = tf.pad(data, [[0, 0], [0, 0], [1, 1], [0, 0]])
    data = tf.nn.separable_conv2d(data, kernel, p, strides=[1, 1, 1, 1], padding='VALID')
    return data


def DownScale(img):
    # return img_down, detail(D, H, V)
    h_psy = tf.constant([[1 / 2, -1 / 2]], dtype='float32')
    h_psy = tf.expand_dims(tf.stack([h_psy, h_psy, h_psy], axis=2), axis=3)  # (1, 2, 3, 1)
    h_phi = tf.constant([[1 / 2, 1 / 2]], dtype='float32')
    h_phi = tf.expand_dims(tf.stack([h_phi, h_phi, h_phi], axis=2), axis=3)  # (1, 2, 3, 1)
    kernel = tf.concat([h_psy, h_phi], axis=2)  # (1, 2, 6, 1)
    p = tf.expand_dims(tf.expand_dims(tf.eye(6), axis=0), axis=0)  # (1, 1, 6, 6)
    data = tf.concat([img, img], axis=3)  # (B, H, W, 2C)
    data = tf.pad(data, [[0, 0], [0, 0], [1, 1], [0, 0]])  # (B, H, W+2, 2C)
    data = tf.nn.separable_conv2d(data, kernel, p, strides=[1, 1, 1, 1], padding='VALID')  # (B, H, W+1, 2C)
    data = DownSample(data, axis=2)  # (B, H, W/2, 2C)
    data = tf.concat([data, data], axis=3)  # (B, H, W/2, 4C)

    h_psy = tf.transpose(h_psy, perm=[1, 0, 2, 3])
    h_phi = tf.transpose(h_phi, perm=[1, 0, 2, 3])
    kernel = tf.concat([h_psy, h_psy, h_phi, h_phi], axis=2)  # (2, 1, 12, 1)
    p = tf.expand_dims(tf.expand_dims(tf.eye(12), axis=0), axis=0)  # (1, 1, 12, 12)
    data = tf.pad(data, [[0, 0], [1, 1], [0, 0], [0, 0]])  # (B, H+2, W/2, 4C)
    data = tf.nn.separable_conv2d(data, kernel, p, strides=[1, 1, 1, 1], padding='VALID')  # (B, H+1, W/2, 4C)
    data = DownSample(data, axis=1)  # (B, H/2, W/2, 4C)
    return data[:, :, :, 9:12], data[:, :, :, 0:9]

