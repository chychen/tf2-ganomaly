import time
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras import layers
import metrics
from absl import logging


class Conv_BN_Act(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 ks,
                 act_type,
                 is_bn=True,
                 padding='same',
                 strides=1,
                 conv_tran=False):
        super(Conv_BN_Act, self).__init__()
        if conv_tran:
            self.conv = layers.Conv2DTranspose(filters,
                                               ks,
                                               strides=strides,
                                               padding=padding,
                                               use_bias=False)
        else:
            self.conv = layers.Conv2D(filters,
                                      ks,
                                      strides=strides,
                                      padding=padding,
                                      use_bias=False)

        self.is_bn = is_bn
        if is_bn:
            self.bn = layers.BatchNormalization(epsilon=1e-05, momentum=0.9)

        if act_type == 'LeakyReLU':
            self.act = layers.LeakyReLU(alpha=0.2)
            self.erase_act = False
        elif act_type == 'ReLU':
            self.act = layers.ReLU()
            self.erase_act = False
        elif act_type == 'Tanh':
            self.act = layers.Activation(tf.tanh)
            self.erase_act = False
        elif act_type == '':
            self.erase_act = True
        else:
            raise ValueError

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.is_bn else x
        x = x if self.erase_act else self.act(x)
        return x


class Encoder(tf.keras.layers.Layer):
    """ DCGAN ENCODER NETWORK
    """
    def __init__(self,
                 isize,
                 nz,
                 nc,
                 ndf,
                 n_extra_layers=0,
                 output_features=False):
        """
        Params:
            isize(int): input image size
            nz(int): num of latent dims
            nc(int): num of input dims
            ndf(int): num of discriminator(Encoder) filters
        """
        super(Encoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        self.in_block = Conv_BN_Act(filters=ndf,
                                    ks=4,
                                    act_type='LeakyReLU',
                                    is_bn=False,
                                    strides=2)
        csize, cndf = isize / 2, ndf

        self.extra_blocks = []
        for t in range(n_extra_layers):
            extra = Conv_BN_Act(filters=cndf, ks=3, act_type='LeakyReLU')
            self.extra_blocks.append(extra)

        self.body_blocks = []
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            body = Conv_BN_Act(filters=out_feat,
                               ks=4,
                               act_type='LeakyReLU',
                               strides=2)
            self.body_blocks.append(body)
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        self.output_features = output_features
        self.out_conv = layers.Conv2D(filters=nz,
                                      kernel_size=4,
                                      padding='valid')

    def call(self, x):
        x = self.in_block(x)
        for block in self.extra_blocks:
            x = block(x)
        for block in self.body_blocks:
            x = block(x)
        last_features = x
        out = self.out_conv(last_features)
        if self.output_features:
            return out, last_features
        else:
            return out


class Decoder(tf.keras.layers.Layer):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        """
        Params:
            isize(int): input image size
            nz(int): num of latent dims
            nc(int): num of input dims
            ngf(int): num of Generator(Decoder) filters
        """
        super(Decoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        self.in_block = Conv_BN_Act(filters=cngf,
                                    ks=4,
                                    act_type='ReLU',
                                    padding='valid',
                                    conv_tran=True)

        csize, _ = 4, cngf
        self.body_blocks = []
        while csize < isize // 2:
            body = Conv_BN_Act(filters=cngf // 2,
                               ks=4,
                               act_type='ReLU',
                               strides=2,
                               conv_tran=True)
            self.body_blocks.append(body)
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        self.extra_blocks = []
        for t in range(n_extra_layers):
            extra = Conv_BN_Act(filters=cngf,
                                ks=3,
                                act_type='ReLU',
                                conv_tran=True)
            self.extra_blocks.append(extra)

        self.out_block = Conv_BN_Act(filters=nc,
                                     ks=4,
                                     act_type='Tanh',
                                     strides=2,
                                     is_bn=False,
                                     conv_tran=True)

    def call(self, x):
        x = self.in_block(x)
        for block in self.body_blocks:
            x = block(x)
        for block in self.extra_blocks:
            x = block(x)
        x = self.out_block(x)
        return x


class NetG(tf.keras.Model):
    def __init__(self, opt):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf,
                                opt.extralayers)
        self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf,
                               opt.extralayers)
        self.encoder2 = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf,
                                opt.extralayers)

    def call(self, x):
        latent_i = self.encoder1(x)
        gen_img = self.decoder(latent_i)
        latent_o = self.encoder2(gen_img)
        return latent_i, gen_img, latent_o

    def num_params(self):
        return sum(
            [np.prod(var.shape.as_list()) for var in self.trainable_variables])


class NetD(tf.keras.Model):
    """ DISCRIMINATOR NETWORK
    """
    def __init__(self, opt):
        super(NetD, self).__init__()
        self.encoder = Encoder(opt.isize,
                               1,
                               opt.nc,
                               opt.ngf,
                               opt.extralayers,
                               output_features=True)
        self.sigmoid = layers.Activation(tf.sigmoid)

    def call(self, x):
        output, last_features = self.encoder(x)
        output = self.sigmoid(output)
        return output, last_features


class GANRunner:
    def __init__(self,
                 G,
                 D,
                 best_state_key,
                 best_state_policy,
                 train_dataset,
                 valid_dataset=None,
                 test_dataset=None,
                 save_path='ckpt/'):
        self.G = G
        self.D = D
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.num_ele_train = self._get_num_element(self.train_dataset)
        self.best_state_key = best_state_key
        self.best_state_policy = best_state_policy
        self.best_state = 1e-9 if self.best_state_policy == max else 1e9
        self.save_path = save_path

    def train_step(self, x, y):
        raise NotImplementedError

    def validate_step(self, x, y):
        raise NotImplementedError

    def evaluate(self, x):
        raise NotImplementedError

    def _get_num_element(self, dataset):
        num_elements = 0
        for _ in dataset:
            num_elements += 1
        return num_elements

    def fit(self, num_epoch, best_state_ths=None):
        self.best_state = self.best_state_policy(
            self.best_state,
            best_state_ths) if best_state_ths is not None else self.best_state
        for epoch in range(num_epoch):
            start_time = time.time()
            # train one epoch
            G_losses = []
            D_losses = []
            with tqdm(total=self.num_ele_train, leave=False) as pbar:
                for step, (x_batch_train,
                           y_batch_train) in enumerate(self.train_dataset):
                    loss = self.train_step(x_batch_train, y_batch_train)
                    G_losses.append(loss[0].numpy())
                    D_losses.append(loss[1].numpy())
                    pbar.update(1)
                G_losses = np.array(G_losses).mean()
                D_losses = np.array(D_losses).mean()
                speed = step * len(x_batch_train) / (time.time() - start_time)
                logging.info(
                    'epoch: {}, G_losses: {:.4f}, D_losses: {:.4f}, samples/sec: {:.4f}'
                    .format(epoch, G_losses, D_losses, speed))

            # validate one epoch
            if self.valid_dataset is not None:
                G_losses = []
                D_losses = []
                for step, (x_batch_train,
                           y_batch_train) in enumerate(self.valid_dataset):
                    loss = self.validate_step(x_batch_train, y_batch_train)
                    G_losses.append(loss[0].numpy())
                    D_losses.append(loss[1].numpy())
                G_losses = np.array(G_losses).mean()
                D_losses = np.array(D_losses).mean()
                logging.info(
                    '\t Validating: G_losses: {}, D_losses: {}'.format(
                        G_losses, D_losses))

            # evaluate on test_dataset
            if self.test_dataset is not None:
                dict_ = self.evaluate(self.test_dataset)
                log_str = '\t Testing:'
                for k, v in dict_.items():
                    log_str = log_str + '   {}: {:.4f}'.format(k, v)
                state_value = dict_[self.best_state_key]
                self.best_state = self.best_state_policy(
                    self.best_state, state_value)
                if self.best_state == state_value:
                    log_str = '*** ' + log_str + ' ***'
                    self.save_best()
                logging.info(log_str)

    def save(self, path):
        self.G.save_weights(self.save_path + 'G')
        self.D.save_weights(self.save_path + 'D')

    def load(self, path):
        self.G.load_weights(self.save_path + 'G')
        self.D.load_weights(self.save_path + 'D')

    def save_best(self):
        self.save(self.save_path + 'best')

    def load_best(self):
        self.load(self.save_path + 'best')


class GANomaly(GANRunner):
    def __init__(self,
                 opt,
                 train_dataset,
                 valid_dataset=None,
                 test_dataset=None):
        self.opt = opt
        self.G = NetG(self.opt)
        self.D = NetD(self.opt)
        super(GANomaly, self).__init__(self.G,
                                       self.D,
                                       best_state_key='roc_auc',
                                       best_state_policy=max,
                                       train_dataset=train_dataset,
                                       valid_dataset=valid_dataset,
                                       test_dataset=test_dataset)
        self.D(tf.keras.Input(shape=[opt.isize, opt.isize, opt.nc]))
        self.D_init_w_path = '/tmp/D_init'
        self.D.save_weights(self.D_init_w_path)

        # label
        self.real_label = tf.ones([
            self.opt.batch_size,
        ], dtype=tf.float32)
        self.fake_label = tf.zeros([
            self.opt.batch_size,
        ], dtype=tf.float32)

        # loss
        l2_loss = tf.keras.losses.MeanSquaredError()
        l1_loss = tf.keras.losses.MeanAbsoluteError()
        bce_loss = tf.keras.losses.BinaryCrossentropy()

        # optimizer
        self.d_optimizer = tf.keras.optimizers.Adam(self.opt.lr,
                                                    beta_1=self.opt.beta1,
                                                    beta_2=0.999)
        self.g_optimizer = tf.keras.optimizers.Adam(self.opt.lr,
                                                    beta_1=self.opt.beta1,
                                                    beta_2=0.999)

        # adversarial loss (use feature matching)
        self.l_adv = l2_loss
        # contextual loss
        self.l_con = l1_loss
        # Encoder loss
        self.l_enc = l2_loss
        # discriminator loss
        self.l_bce = bce_loss

    def _evaluate(self, test_dataset):
        an_scores = []
        gt_labels = []
        for step, (x_batch_train, y_batch_train) in enumerate(test_dataset):
            latent_i, gen_img, latent_o = self.G(x_batch_train)
            latent_i, gen_img, latent_o = latent_i.numpy(), gen_img.numpy(
            ), latent_o.numpy()
            error = np.mean((latent_i - latent_o)**2, axis=-1)
            an_scores.append(error)
            gt_labels.append(y_batch_train)
        an_scores = np.concatenate(an_scores, axis=0).reshape([-1])
        gt_labels = np.concatenate(gt_labels, axis=0).reshape([-1])
        return an_scores, gt_labels

    def evaluate(self, test_dataset):
        ret_dict = {}
        an_scores, gt_labels = self._evaluate(test_dataset)
        # normed to [0,1)
        an_scores = (an_scores - np.amin(an_scores)) / (np.amax(an_scores) -
                                                        np.amin(an_scores))
        # AUC
        auc_dict = metrics.roc_auc(gt_labels, an_scores)
        ret_dict.update(auc_dict)
        # Average Precision
        p_r_dict = metrics.pre_rec_curve(gt_labels, an_scores)
        ret_dict.update(p_r_dict)
        return ret_dict

    def evaluate_best(self, test_dataset):
        self.load_best()
        an_scores, gt_labels = self._evaluate(test_dataset)
        # AUC
        _ = metrics.roc_auc(gt_labels, an_scores, show=True)
        # Average Precision
        _ = metrics.pre_rec_curve(gt_labels, an_scores, show=True)

    @tf.function
    def _train_step_autograph(self, x):
        """ Autograph enabled by tf.function could speedup more than 6x than eager mode.
        """
        self.input = x
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            self.latent_i, self.gen_img, self.latent_o = self.G(self.input)
            self.pred_real, self.feat_real = self.D(self.input)
            self.pred_fake, self.feat_fake = self.D(self.gen_img)
            g_loss = self.g_loss()
            d_loss = self.d_loss()

        g_grads = g_tape.gradient(g_loss, self.G.trainable_weights)
        d_grads = d_tape.gradient(d_loss, self.D.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads,
                                             self.G.trainable_weights))
        self.d_optimizer.apply_gradients(zip(d_grads,
                                             self.D.trainable_weights))
        return g_loss, d_loss

    def train_step(self, x, y):
        g_loss, d_loss = self._train_step_autograph(x)
        if d_loss < 1e-5:
            st = time.time()
            self.D.load_weights(self.D_init_w_path)
            logging.info('re-init D, cost: {:.4f} secs'.format(time.time() -
                                                               st))

        return g_loss, d_loss

    def validate_step(self, x, y):
        pass

    def g_loss(self):
        self.err_g_adv = self.l_adv(self.feat_real, self.feat_fake)
        self.err_g_con = self.l_con(self.input, self.gen_img)
        self.err_g_enc = self.l_enc(self.latent_i, self.latent_o)
        g_loss = self.err_g_adv * self.opt.w_adv + \
                self.err_g_con * self.opt.w_con + \
                self.err_g_enc * self.opt.w_enc
        return g_loss

    def d_loss(self):
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)
        d_loss = (self.err_d_real + self.err_d_fake) * 0.5
        return d_loss
