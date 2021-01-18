import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
import os
import time
import numpy as np
import cv2
from model import GANomaly

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_integer("shuffle_buffer_size", 10000,
                     "buffer size for pseudo shuffle")
flags.DEFINE_integer("batch_size", 300, "batch_size")
flags.DEFINE_integer("isize", None, "input size")
flags.DEFINE_string("ckpt_dir", 'ckpt', "checkpoint folder")
flags.DEFINE_integer("nz", 100, "latent dims")
flags.DEFINE_integer("nc", None, "input channels")
flags.DEFINE_integer("ndf", 64, "number of discriminator's filters")
flags.DEFINE_integer("ngf", 64, "number of generator's filters")
flags.DEFINE_integer("extralayers", 0, "extralayers for both G and D")
flags.DEFINE_list("encdims", None, "Layer dimensions of the encoder and in reverse of the decoder."
                                   "If given, dense encoder and decoders are used.")
flags.DEFINE_integer("niter", 15, "number of training epochs")
flags.DEFINE_float("lr", 2e-4, "learning rate")
flags.DEFINE_float("w_adv", 1., "Adversarial loss weight")
flags.DEFINE_float("w_con", 50., "Reconstruction loss weight")
flags.DEFINE_float("w_enc", 1., "Encoder loss weight")
flags.DEFINE_float("beta1", 0.5, "beta1 for Adam optimizer")
flags.DEFINE_string("dataset", None, "name of dataset")
DATASETS = ['mnist', 'cifar10']
flags.register_validator('dataset',
                         lambda name: name in DATASETS,
                         message='--dataset must be {}'.format(DATASETS))
flags.DEFINE_integer("anomaly", None, "the anomaly idx")
flags.mark_flag_as_required('anomaly')
flags.mark_flag_as_required('isize')
flags.mark_flag_as_required('nc')

def batch_resize(imgs, size: tuple):
    img_out = np.empty((imgs.shape[0], ) + size)
    for i in range(imgs.shape[0]):
        img_out[i] = cv2.resize(imgs[i], size, interpolation=cv2.INTER_CUBIC)
    return img_out


def main(_):
    opt = FLAGS
    # logging
    logging.set_verbosity(logging.INFO)
    logging.set_stderrthreshold(logging.INFO)
    if FLAGS.log_dir:
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        logging.get_absl_handler().use_absl_log_file(FLAGS.dataset, log_dir=FLAGS.log_dir)
    # dataset
    if opt.dataset=='mnist':
        data_train, data_test = tf.keras.datasets.mnist.load_data()
    elif opt.dataset=='cifar10':
        data_train, data_test = tf.keras.datasets.cifar10.load_data()
    else:
        raise NotImplementError
    x_train, y_train = data_train
    x_test, y_test = data_test
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.reshape([-1,])
    y_test = y_test.reshape([-1,])
    # resize to (32, 32)
    if opt.dataset=='mnist':
        x_train = batch_resize(x_train, (32, 32))[..., None]
        x_test = batch_resize(x_test, (32, 32))[..., None]
    # normalization
    mean = x_train.mean()
    stddev = x_train.std()
    x_train = (x_train - mean) / stddev
    x_test = (x_test - mean) / stddev
    logging.info('{}, {}'.format(x_train.shape, x_test.shape))
    # define abnoraml data and normal
    # training data only contains normal
    x_train = x_train[y_train != opt.anomaly, ...]
    y_train = y_train[y_train != opt.anomaly, ...]
    y_test = (y_test == opt.anomaly).astype(np.float32)
    # tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.shuffle(opt.shuffle_buffer_size).batch(
        opt.batch_size, drop_remainder=True)
    test_dataset = test_dataset.batch(opt.batch_size, drop_remainder=False)

    # training
    ganomaly = GANomaly(opt,
                        train_dataset,
                        valid_dataset=None,
                        test_dataset=test_dataset)
    ganomaly.fit(opt.niter)

    # evaluating
    ganomaly.evaluate_best(test_dataset)


if __name__ == '__main__':
    app.run(main)
