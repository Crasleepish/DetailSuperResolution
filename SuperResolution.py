import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from PIL import Image
import datetime
import pathlib
import pickle
import sys

# tf.debugging.set_log_device_placement(True)
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)


from config import *
from FirstModel import *
from WaveletTrans import *
import ModelFactory as mf


# Reference:
# Photo-realistic single image super-resolution using a generative adversarial network. CVPR 2017
# Esrgan: Enhanced super-resolution generative adversarial networks. ECCV 2018


vgg_model = VGG19(weights='imagenet', include_top=False)
vgg_model.trainable = False

feature_extractor_5_4 = tf.keras.models.Model(inputs=vgg_model.input,
                                              outputs=vgg_model.get_layer('block5_conv4').output)
feature_extractor_5_4.trainable = False

first_model = get_first_model()

summary_joint_writer = tf.summary.create_file_writer(
    log_dir + "detailgan/experiment12")
down2_writer = tf.summary.create_file_writer(
    log_dir + "detailgan/experiment12_BILINEAR")
down3_writer = tf.summary.create_file_writer(
    log_dir + "detailgan/experiment12_LANCZOS3")
down4_writer = tf.summary.create_file_writer(
    log_dir + "detailgan/experiment12_OVERDOWN")
down_writes = [down2_writer, down3_writer, down4_writer]

try:
    with open(checkpoint_dir + '/status.pickle', 'rb') as f:
        status = pickle.load(f)
except Exception:
    with open(checkpoint_dir + '/status.pickle', 'wb') as f:
        status = {'n_steps': 0}
        pickle.dump(status, f)
n_steps = status['n_steps']

dFactory = mf.ModelFactory()


def preprocess_img(filename):
    img = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img)
    # project pixel value into [-1, 1]
    img = (tf.cast(img, 'float32') - 127.5) / 127.5
    return img


def detail_decompose(x):
    approx, detail1 = DownScale(x)
    decomposed = tf.concat([detail1, approx], axis=3)
    return decomposed


def feature_extractor(x):
    x = x * 127.5 + 127.5
    x = bicubicResize(x, [256, 256], [224, 224], 0., 255.)
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    _IMAGENET_MEAN = tf.constant(-np.array(mean), dtype='float32')
    x = tf.nn.bias_add(x, _IMAGENET_MEAN)
    return feature_extractor_5_4(x)


def write_joint_logs(d_loss, joint_loss, valid_loss, _psnr, _ssim, _mv, n_steps):
    with summary_joint_writer.as_default():
        tf.summary.scalar('disc_loss', d_loss, step=n_steps)
        tf.summary.scalar('gen_loss', joint_loss, step=n_steps)
        tf.summary.scalar('valid_loss', valid_loss, step=n_steps)
        tf.summary.scalar('psnr_of_valid_imgs', _psnr, step=n_steps)
        tf.summary.scalar('ssim_of_valid_imgs', _ssim, step=n_steps)
        tf.summary.scalar('meanVariance_of_valid_imgs', _mv, step=n_steps)
    record_exp_data(n_steps, valid_loss, _psnr, _ssim, _mv)


def write_data_of_unknowndown(ukdown_datas, n_steps):
    for item in zip(down_writes, ukdown_datas):
        writer = item[0]
        with writer.as_default():
            tf.summary.scalar('psnr_of_valid_imgs', item[1][0], step=n_steps)
            tf.summary.scalar('ssim_of_valid_imgs', item[1][1], step=n_steps)
            tf.summary.scalar('meanVariance_of_valid_imgs', item[1][2], step=n_steps)
    record_ukdown_data(n_steps, ukdown_datas)


def record_exp_data(_steps, _valid_loss, _psnr, _ssim, _mv):
    if isinstance(_steps, tf.Tensor):
        _steps = _steps.numpy()
    if isinstance(_valid_loss, tf.Tensor):
        _valid_loss = _valid_loss.numpy()
    if isinstance(_psnr, tf.Tensor):
        _psnr = _psnr.numpy()
    if isinstance(_ssim, tf.Tensor):
        _ssim = _ssim.numpy()
    if isinstance(_mv, tf.Tensor):
        _mv = _mv.numpy()
    with open(checkpoint_dir + '/exp_data.txt', 'a') as f:
        f.write(','.join([str(_steps), str(_valid_loss), str(_psnr), str(_ssim), str(_mv)]))
        f.write('\n')


def record_ukdown_data(_steps, ukdown_datas):
    f2 = open(checkpoint_dir + '/down2_data.txt', 'a')
    f3 = open(checkpoint_dir + '/down3_data.txt', 'a')
    f4 = open(checkpoint_dir + '/down4_data.txt', 'a')
    fs = [f2, f3, f4]
    for item in zip(fs, ukdown_datas):
        _f = item[0]
        _psnr = item[1][0]
        _ssim = item[1][1]
        _mv = item[1][2]
        if isinstance(_psnr, tf.Tensor):
            _psnr = _psnr.numpy()
        if isinstance(_ssim, tf.Tensor):
            _ssim = _ssim.numpy()
        if isinstance(_mv, tf.Tensor):
            _mv = _mv.numpy()
        _f.write(','.join([str(_steps), str(_psnr), str(_ssim), str(_mv)]))
        _f.write('\n')
        _f.close()


class TrainPhase:
    def __init__(self, phase, first_model):
        self.phase = phase
        self.first_model = first_model
        self.d = dFactory.get_d_model()
        self.g_optimizer = tf.keras.optimizers.Adam(GAN_LR, beta_1=0.5, beta_2=0.999)
        self.d_optimizer = tf.keras.optimizers.Adam(GAN_LR, beta_1=0.5, beta_2=0.999)
        self.ckpt = tf.train.Checkpoint(first_model=self.first_model,
                                        d_optimizer=self.d_optimizer,
                                        g_optimizer=self.g_optimizer,
                                        d=self.d)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir + '/phase_{}'.format(self.phase), max_to_keep=5)
        self.downLossWeight = DOWNLOSS_WEIGHT
        self.detailWeight = DETAIL_WEIGHT
        if self.phase == 0:
            self.detailWeight = 0.

    def first_loss(self, realX, predictX):
        return tf.reduce_mean(
            tf.square(feature_extractor(realX) - feature_extractor(predictX))
        )

    def down_loss(self, down, predictX, r):
        p_down = blurDown(predictX, [256, 256], [64, 64], r)
        return tf.reduce_mean(
            tf.square(down - p_down)
        )

    def _f1(self, x):
        return tf.square(x - 1.)

    def _f2(self, x):
        return tf.square(x + 1.)

    # relativistic discriminator: https://arxiv.org/pdf/1807.00734
    def discriminator_loss(self, realX, fakeX):
        cr_bar = tf.reduce_mean(self.d(realX), keepdims=True)
        cf_bar = tf.reduce_mean(self.d(fakeX), keepdims=True)
        return self.detailWeight * tf.reduce_mean(
            self._f1(self.d(realX) - cf_bar) + self._f2(self.d(fakeX) - cr_bar)
        )

    # relativistic discriminator: https://arxiv.org/pdf/1807.00734
    def generator_loss(self, realX, fakeX):
        cr_bar = tf.reduce_mean(self.d(realX), keepdims=True)
        cf_bar = tf.reduce_mean(self.d(fakeX), keepdims=True)
        return self.detailWeight * tf.reduce_mean(
            self._f1(self.d(fakeX) - cr_bar) + self._f2(self.d(realX) - cf_bar)
        )

    def joint_loss(self, x1, x2, predict, down, r):
        l1 = self.first_loss(x1, predict)
        dl = self.down_loss(down, predict, r)
        gl = self.generator_loss(detail_decompose(x2), detail_decompose(predict))
        return l1 + self.downLossWeight * dl + gl

    @tf.function
    def g_train_step(self, x1, x2):
        r = getRandomRadius()
        down = blurDown(x1, [256, 256], [64, 64], r)
        with tf.GradientTape(watch_accessed_variables=False) as gen_tape:
            gen_tape.watch(self.first_model.trainable_variables)
            predict = self.first_model(down)
            j_loss = self.joint_loss(x1, x2, predict, down, r)
        gradient_g = gen_tape.gradient(j_loss, self.first_model.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradient_g, self.first_model.trainable_variables))
        return j_loss

    @tf.function
    def d_train_step(self, x1, x2):
        r = getRandomRadius()
        down = blurDown(x1, [256, 256], [64, 64], r)
        predict = self.first_model(down)
        with tf.GradientTape(watch_accessed_variables=False) as disc_tape:
            disc_tape.watch(self.d.trainable_variables)
            d_loss = self.discriminator_loss(detail_decompose(x2), detail_decompose(predict))
        gradient_d = disc_tape.gradient(d_loss, self.d.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradient_d, self.d.trainable_variables))
        return d_loss

    @tf.function
    def compute_valid_loss(self, x):
        r = getRandomRadius()
        y = blurDown(x, [256, 256], [64, 64], r)
        y = self.first_model(y)
        valid_loss = self.first_loss(x, y)
        return valid_loss


def joint_train(train_dataset, valid_dataset, epochs):
    global n_steps
    bar = ProgressBar(n_steps)
    # decide starting phase
    phase = 1
    K.clear_session()
    trainPhase = TrainPhase(phase, first_model)
    print("Current steps: {}, phase: {}".format(n_steps, phase))
    # load checkpoint
    if trainPhase.ckpt_manager.latest_checkpoint:
        trainPhase.ckpt.restore(trainPhase.ckpt_manager.latest_checkpoint)
        print("Restored from {}".format(trainPhase.ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    for epoch in range(epochs):
        if trainPhase.g_optimizer.learning_rate != GAN_LR:
            trainPhase.g_optimizer.learning_rate.assign(GAN_LR)
        if trainPhase.d_optimizer.learning_rate != GAN_LR:
            trainPhase.d_optimizer.learning_rate.assign(GAN_LR)
        print("epoch: {}".format(epoch + 1))
        print("Learning Rate: {}".format(trainPhase.d_optimizer.learning_rate))
        starttime = datetime.datetime.now().timestamp()
        it = iter(train_dataset)
        v_it = iter(valid_dataset)
        has_next = True
        while has_next:
            d_loss = 0.
            for _ in range(N_CRITIC):
                try:
                    x1 = next(it)
                    x2 = next(it)
                    d_loss = trainPhase.d_train_step(x1, x2)
                except StopIteration:
                    has_next = False
                    d_loss = 0.
                    break

            try:
                x1 = next(it)
                x2 = next(it)
                j_loss = trainPhase.g_train_step(x1, x2)
            except StopIteration:
                has_next = False
                j_loss = 0.

            n_steps += 1
            bar.step_forward()
            if n_steps % 100 == 0:
                try:
                    x = next(v_it)
                except StopIteration:
                    v_it = iter(valid_dataset)
                    x = next(v_it)
                valid_loss = trainPhase.compute_valid_loss(x)
                _psnr, _ssim, _mv, ukdown_datas = compute_valid_psnr_ssim_mv(valid_hr_imgs)
                write_joint_logs(d_loss, j_loss, valid_loss, _psnr, _ssim, _mv, n_steps // 100)
                write_data_of_unknowndown(ukdown_datas, n_steps // 100)
            if n_steps % 1000 == 0:
                show_sample_image(train_imgs, n_steps // 1000)
                show_valid_image(valid_hr_imgs, n_steps // 1000)
                saveckpt(trainPhase.ckpt_manager)

        print('\n', end='')
        endtime = datetime.datetime.now().timestamp()
        print("epoch {} end. take {} seconds.".format(epoch + 1, endtime - starttime))
        saveckpt(trainPhase.ckpt_manager)


def saveckpt(ckptmanager):
    ckptmanager.save()
    with open(checkpoint_dir + '/status.pickle', 'wb') as f:
        status = {'n_steps': n_steps}
        pickle.dump(status, f)


def show_sample_image(imgs, index):
    imgs = tf.stack([preprocess_img(f) for f in imgs])
    r = getRandomRadius()
    test_imgs_down = blurDown(imgs, [256, 256], [64, 64], r)
    predict_hr = first_model(test_imgs_down)
    real_hr = imgs
    test_imgs_down = K.clip(tf.image.resize(test_imgs_down, [256, 256], method=tf.image.ResizeMethod.BICUBIC, antialias=True), -1, 1)
    # show images
    plt.close()
    fig = plt.figure(figsize=(10, 20), dpi=100)
    n = imgs.shape[0]
    # create a empty saved image
    outImg = Image.new('RGB', (808, 1606))
    for i in range(n):
        plt.subplot(n, 3, i * 3 + 1)
        x = K.round(test_imgs_down[i] * 127.5 + 127.5).numpy().astype('uint8')
        plt.imshow(x)
        plt.axis('off')
        img = Image.fromarray(x)
        outImg.paste(img, (10, 10 + i * 266))

        plt.subplot(n, 3, i * 3 + 2)
        x = K.round(predict_hr[i] * 127.5 + 127.5).numpy().astype('uint8')
        plt.imshow(x)
        plt.axis('off')
        img = Image.fromarray(x)
        outImg.paste(img, (10 + 266 * 1, 10 + i * 266))

        plt.subplot(n, 3, i * 3 + 3)
        x = K.round(real_hr[i] * 127.5 + 127.5).numpy().astype('uint8')
        plt.imshow(x)
        plt.axis('off')
        img = Image.fromarray(x)
        outImg.paste(img, (10 + 266 * 2, 10 + i * 266))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()
    outImg.save(checkpoint_dir + '/step_{}.png'.format(index))


def show_valid_image(imgs, index):
    real_hr = tf.stack([preprocess_img(f) for f in imgs])
    # BICUBIC
    down1 = blurDown(real_hr, [256, 256], [64, 64], 0)
    # BILINEAR
    down2 = K.clip(tf.image.resize(real_hr, [64, 64], method=tf.image.ResizeMethod.BILINEAR, antialias=True), -1., 1.)
    # LANCZOS3
    down3 = K.clip(tf.image.resize(real_hr, [64, 64], method=tf.image.ResizeMethod.LANCZOS3, antialias=True), -1., 1.)
    # OVERDOWN
    down4 = K.clip(tf.image.resize(real_hr, [60, 60], method=tf.image.ResizeMethod.BICUBIC, antialias=True), -1., 1.)
    down4 = K.clip(tf.image.resize(down4, [64, 64], method=tf.image.ResizeMethod.BICUBIC, antialias=True), -1., 1.)
    downs = [down1, down2, down3, down4]

    # show images
    plt.close()
    plt.figure(figsize=(40, 20), dpi=100)
    # create a empty saved image
    outImg = Image.new('RGB', (2404, 1606))
    for k in range(len(downs)):
        down = downs[k]
        n = down.shape[0]
        predict_hr = first_model(down)
        down = K.clip(tf.image.resize(down, [256, 256], method=tf.image.ResizeMethod.BICUBIC, antialias=True), -1, 1)
        for i in range(n):
            plt.subplot(n, 9, 9 * i + 2 * k + 1)
            x = K.round(down[i] * 127.5 + 127.5).numpy().astype('uint8')
            plt.imshow(x)
            plt.axis('off')
            img = Image.fromarray(x)
            outImg.paste(img, (10 + 2 * k * 266, 10 + i * 266))

            plt.subplot(n, 9, 9 * i + 2 * k + 2)
            x = K.round(predict_hr[i] * 127.5 + 127.5).numpy().astype('uint8')
            plt.imshow(x)
            plt.axis('off')
            img = Image.fromarray(x)
            outImg.paste(img, (10 + (2 * k + 1) * 266, 10 + i * 266))

    n = real_hr.shape[0]
    for i in range(n):
        plt.subplot(n, 9, (i + 1) * 9)
        x = K.round(real_hr[i] * 127.5 + 127.5).numpy().astype('uint8')
        plt.imshow(x)
        plt.axis('off')
        img = Image.fromarray(x)
        outImg.paste(img, (10 + 266 * 8, 10 + i * 266))

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()
    outImg.save(checkpoint_dir + '/step_v_{}.png'.format(index))


# Wang Z, Bovik A C, Sheikh H R, et al.
# Image quality assessment: from error visibility to structural similarity[J].
# IEEE transactions on image processing, 2004, 13(4): 600-612.
def compute_valid_psnr_ssim_mv(imgs):
    real_hr = tf.stack([preprocess_img(f) for f in imgs])
    down = blurDown(real_hr, [256, 256], [64, 64], 0)
    predict_hr = first_model(down)
    real_hr_255 = K.round(real_hr * 127.5 + 127.5)
    predict_hr_255 = K.round(predict_hr * 127.5 + 127.5)
    _psnr = psnr(predict_hr_255, real_hr_255)
    _ssim = ssim(predict_hr_255, real_hr_255)
    _meanvar = meanVar(predict_hr_255)

    # unknown down
    ukdown_datas = []
    # BILINEAR
    down2 = K.clip(tf.image.resize(real_hr, [64, 64], method=tf.image.ResizeMethod.BILINEAR, antialias=True), -1., 1.)
    # LANCZOS3
    down3 = K.clip(tf.image.resize(real_hr, [64, 64], method=tf.image.ResizeMethod.LANCZOS3, antialias=True), -1., 1.)
    # OVERDOWN
    down4 = K.clip(tf.image.resize(real_hr, [60, 60], method=tf.image.ResizeMethod.BICUBIC, antialias=True), -1., 1.)
    down4 = K.clip(tf.image.resize(down4, [64, 64], method=tf.image.ResizeMethod.BICUBIC, antialias=True), -1., 1.)
    downs = [down2, down3, down4]
    for _down in downs:
        predict_hr = first_model(_down)
        predict_hr_255 = K.round(predict_hr * 127.5 + 127.5)
        metric1 = tf.reduce_mean(psnr(predict_hr_255, real_hr_255))
        metric2 = tf.reduce_mean(ssim(predict_hr_255, real_hr_255))
        metric3 = meanVar(predict_hr_255)
        ukdown_datas.append((metric1, metric2, metric3))
    return tf.reduce_mean(_psnr), tf.reduce_mean(_ssim), _meanvar, ukdown_datas


class ProgressBar:
    def __init__(self, steps):
        self.steps = steps
        self.toolbar_width = 100

    def printBar(self):
        c = self.steps % 1000 * self.toolbar_width // 1000
        sys.stdout.write('\r')
        sys.stdout.write('[%s%s]' % ('>' * c, '.' * (self.toolbar_width - c)))
        sys.stdout.write(str(self.steps))
        sys.stdout.flush()

    def step_forward(self):
        self.steps += 1
        self.printBar()


train_imgs = [
    "./datasets/train2014_patch/train/COCO_train2014_000000000009_1.jpg",
    "./datasets/train2014_patch/train/COCO_train2014_000000000061_1.jpg",
    "./datasets/train2014_patch/train/COCO_train2014_000000285064_2.jpg",
    "./datasets/train2014_patch/train/COCO_train2014_000000286234_2.jpg",
    "./datasets/train2014_patch/train/COCO_train2014_000000369637_2.jpg",
    "./datasets/train2014_patch/train/COCO_train2014_000000451206_2.jpg"
]

valid_hr_imgs = [
    "./datasets/train2014_patch/valid/43074_HR.jpg",
    "./datasets/train2014_patch/valid/baboon_HR.jpg",
    "./datasets/train2014_patch/valid/bird_HR.jpg",
    "./datasets/train2014_patch/valid/butterfly_HR.jpg",
    "./datasets/train2014_patch/valid/comic_HR.jpg",
    "./datasets/train2014_patch/valid/woman_HR.jpg"
]

if __name__ == '__main__':
    # patches cut out from COCO train 2014 dataset
    path0 = "./datasets/train2014_patch/train"
    files0 = list(pathlib.Path(path0).glob('*.jpg'))
    files0 = [str(f) for f in files0]

    train_dataset = tf.data.Dataset.from_tensor_slices(files0)
    train_dataset = train_dataset.shuffle(201000)
    val_dataset = train_dataset.take(1000)
    train_dataset = train_dataset.skip(1000)

    train_dataset = train_dataset.map(preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(8192, reshuffle_each_iteration=True).batch(BATCH_SIZE)
    val_dataset = val_dataset.shuffle(1000, reshuffle_each_iteration=True).batch(BATCH_SIZE)

    joint_train(train_dataset, val_dataset, EPOCHS)
