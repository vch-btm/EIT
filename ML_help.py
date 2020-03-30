# coding=utf-8

from __future__ import print_function, division

import sys
import os

from shutil import copyfile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

os.environ["CUDA_VISIBLE_DEVICES"] = "0";

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{sys.argv[1]}";

import time
import keras
from keras import backend as K
import tensorflow as tf

config = tf.ConfigProto(device_count={'GPU': 1})  # , 'CPU': 4})
sess = tf.Session(config=config)
K.set_session(sess)

from keras.layers import Concatenate, Conv2DTranspose, Lambda, Layer, Conv2D, UpSampling2D, BatchNormalization
from keras.engine import InputSpec
from keras.optimizers import Optimizer
# import normalization
# from keras_contrib.layers.normalization import InstanceNormalization
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
# mpl.use('TkAgg')
import matplotlib.pyplot as plt


class Swish(Layer):
    def __init__(self, beta, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return K.sigmoid(self.beta * inputs) * inputs

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Mish(Layer):
    # mish(x) = x*tanh(softplus(x)) = x*tanh(ln(1+e^x))

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Brish(Layer):
    def __init__(self, c1, c2, sigma, **kwargs):
        super(Brish, self).__init__(**kwargs)
        self.c1 = K.cast_to_floatx(c1)
        self.c2 = K.cast_to_floatx(c2)
        self.sigma = K.cast_to_floatx(sigma)

        self.cp = K.cast_to_floatx((self.c1 + self.c2) / 2)
        self.cm = K.cast_to_floatx((self.c1 - self.c2) / 2)
        self.cms = K.cast_to_floatx((self.c1 - self.c2) / np.sqrt(2 * np.pi) * self.sigma)
        self.s1 = K.cast_to_floatx(np.sqrt(2) * self.sigma)
        self.s2 = K.cast_to_floatx(2 * self.sigma ** 2)

    def call(self, inputs):
        return self.cm * inputs * tf.math.erf(inputs / self.s1) + self.cms * K.exp(- K.square(inputs) / self.s2) + self.cp * inputs

    def get_config(self):
        config = {'c1': float(self.c1), 'c2': float(self.c2), 'sigma': float(self.sigma)}
        base_config = super(Brish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, inputs, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(inputs, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class RAdam1(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., **kwargs):
        super(RAdam1, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    # @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)
        rho = 2 / (1 - self.beta_2) - 1
        rho_t = rho - 2 * t * beta_2_t / (1 - beta_2_t)
        r_t = K.sqrt(K.relu(rho_t - 4) * K.relu(rho_t - 2) * rho / ((rho - 4) * (rho - 2) * rho_t))
        flag = K.cast(rho_t > 4, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            mhat_t = m_t / (1 - beta_1_t)
            vhat_t = K.sqrt(v_t / (1 - beta_2_t))
            p_t = p - lr * mhat_t * (flag * r_t / (vhat_t + self.epsilon) + (1 - flag))

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(RAdam1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RAdam2(keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0., amsgrad=False, total_steps=0, warmup_proportion=0.1, min_lr=0., **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super(RAdam2, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.total_steps = K.variable(total_steps, name='total_steps')
            self.warmup_proportion = K.variable(warmup_proportion, name='warmup_proportion')
            self.min_lr = K.variable(min_lr, name='min_lr')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.initial_weight_decay = weight_decay
        self.initial_total_steps = total_steps
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        if self.initial_total_steps > 0:
            warmup_steps = self.total_steps * self.warmup_proportion
            decay_steps = K.maximum(self.total_steps - warmup_steps, 1)
            decay_rate = (self.min_lr - lr) / decay_steps
            lr = K.switch(
                t <= warmup_steps,
                lr * (t / warmup_steps),
                lr + decay_rate * K.minimum(t - warmup_steps, decay_steps),
            )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vhat_' + str(i)) for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))]

        self.weights = [self.iterations] + ms + vs + vhats

        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)

        sma_inf = 2.0 / (1.0 - self.beta_2) - 1.0
        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            m_corr_t = m_t / (1.0 - beta_1_t)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t))
                self.updates.append(K.update(vhat, vhat_t))
            else:
                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t))

            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                         (sma_t - 2.0) / (sma_inf - 2.0) *
                         sma_inf / sma_t)

            p_t = K.switch(sma_t >= 5, r_t * m_corr_t / (v_corr_t + self.epsilon), m_corr_t)

            if self.initial_weight_decay > 0:
                p_t += self.weight_decay * p

            p_t = p - lr * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, learning_rate):
        self.learning_rate = learning_rate

    def get_config(self):
        config = {
            'learning_rate': float(K.get_value(self.learning_rate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': float(K.get_value(self.weight_decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_steps': float(K.get_value(self.total_steps)),
            'warmup_proportion': float(K.get_value(self.warmup_proportion)),
            'min_lr': float(K.get_value(self.min_lr)),
        }
        base_config = super(RAdam2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Lookahead(keras.optimizers.Optimizer):
    """The lookahead mechanism for optimizers.

    Default parameters follow those provided in the original paper.

    # Arguments
        optimizer: An existed optimizer.
        sync_period: int > 0. The synchronization period.
        slow_step: float, 0 < alpha < 1. The step size of slow weights.

    # References
        - [Lookahead Optimizer: k steps forward, 1 step back]
          (https://arxiv.org/pdf/1907.08610v1.pdf)
    """

    def __init__(self, optimizer, sync_period=5, slow_step=0.5, **kwargs):
        super(Lookahead, self).__init__(**kwargs)
        self.optimizer = keras.optimizers.get(optimizer)
        with K.name_scope(self.__class__.__name__):
            self.sync_period = K.variable(sync_period, dtype='int64', name='sync_period')
            self.slow_step = K.variable(slow_step, name='slow_step')

    @property
    def lr(self):
        return self.optimizer.lr

    @lr.setter
    def lr(self, lr):
        self.optimizer.lr = lr

    @property
    def learning_rate(self):
        return self.optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    @property
    def iterations(self):
        return self.optimizer.iterations

    def get_updates(self, loss, params):
        sync_cond = K.equal((self.iterations + 1) // self.sync_period * self.sync_period, (self.iterations + 1))

        slow_params = {p.name: K.variable(K.get_value(p), name=f'sp_{i}') for i, p in enumerate(params)}
        update_names = ['update', 'update_add', 'update_sub']
        original_updates = [getattr(K, name) for name in update_names]
        setattr(K, 'update', lambda x, new_x: ('update', x, new_x))
        setattr(K, 'update_add', lambda x, new_x: ('update_add', x, new_x))
        setattr(K, 'update_sub', lambda x, new_x: ('update_sub', x, new_x))
        self.updates = self.optimizer.get_updates(loss, params)
        for name, original_update in zip(update_names, original_updates):
            setattr(K, name, original_update)
        slow_updates = []
        for i, update in enumerate(self.updates):
            if isinstance(update, tuple):
                name, x, new_x, adjusted = update + (update[-1],)
                update_func = getattr(K, name)
                if name == 'update_add':
                    adjusted = x + new_x
                if name == 'update_sub':
                    adjusted = x - new_x
                if x.name not in slow_params:
                    self.updates[i] = update_func(x, new_x)
                else:
                    slow_param = slow_params[x.name]
                    slow_param_t = slow_param + self.slow_step * (adjusted - slow_param)
                    slow_updates.append(K.update(slow_param, K.switch(
                        sync_cond,
                        slow_param_t,
                        slow_param,
                    )))
                    self.updates[i] = K.update(x, K.switch(
                        sync_cond,
                        slow_param_t,
                        adjusted,
                    ))
        slow_params = list(slow_params.values())
        self.updates += slow_updates
        self.weights = self.optimizer.weights + slow_params
        return self.updates

    def get_config(self):
        config = {
            'optimizer': keras.optimizers.serialize(self.optimizer),
            'sync_period': int(K.get_value(self.sync_period)),
            'slow_step': float(K.get_value(self.slow_step)),
        }
        base_config = super(Lookahead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        optimizer = keras.optimizers.deserialize(config.pop('optimizer'))
        return cls(optimizer, **config)


class LookaheadInject(object):
    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model):
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        model._check_trainable_weights_consistency()

        if model.train_function is None:
            inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
            if model._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]
            fast_params = model._collected_trainable_weights

            with K.name_scope('training'):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(params=fast_params, loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]
                fast_updates = (model.updates + training_updates + model.metrics_updates)

                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(inputs, [model.total_loss] + model.metrics_tensors, updates=fast_updates, name='fast_train_function', **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R

                model.train_function = F


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def start_ML(randomTxt):
    device_name = ""

    my_config = tf.ConfigProto()
    my_config.gpu_options.allow_growth = True
    session = tf.Session(config=my_config)

    if len(sys.argv) > 1:
        device_name = f'/device:GPU:{sys.argv[1]}'
        # randomTxt = sys.argv[2]
    else:
        device_name = tf.test.gpu_device_name()
        device_name = '/device:GPU:0'
        # device_name = '/device:GPU:1'

        if device_name != '/device:GPU:0':
            print(f"device_name = {device_name}")

            raise SystemError('GPU device not found')

    print(f'Found GPU at: {device_name}')
    print(f"keras version: {keras.__version__}")
    print(f"tensorflow version: {tf.__version__}")

    return f"{time.strftime('%Y%m%d_%H%M')}_{randomTxt}"


def get_boundaries(arr):
    arr2 = np.copy(arr)

    if arr2.ndim == 2:
        arr2[slice(1, -1), slice(1, -1)] = 0.0
    else:
        arr2[:, slice(1, -1), slice(1, -1), :] = 0.0
        return arr2


def sample_images(epoch, net, subpathID, A_test, B_test, img_shape, img_rows, img_cols, images_to_sample, v_min=1, v_max=2):
    os.makedirs(f'{subpathID}/imagesAll', exist_ok=True)
    r, c = len(images_to_sample), 5

    fig, axs = plt.subplots(r, c, figsize=(2.75 * c, 2 * r))

    for i in range(r):
        cnt = 0

        img_A = A_test[images_to_sample[i]].reshape((1,) + img_shape)
        img_B = B_test[images_to_sample[i]].reshape((1, img_rows, img_cols, 1))

        fake_B = net.predict(img_A)

        gen_imgs = np.concatenate([img_B, fake_B, fake_B, img_B - fake_B, img_B - fake_B])

        for j in range(c):
            ax = axs[i, j]

            switcher = {
                0: {'vmin': v_min, 'vmax': v_max},
                1: {'vmin': v_min, 'vmax': v_max},
                3: {'vmin': -0.05, 'vmax': 0.05, 'cmap': plt.get_cmap("seismic")},
                4: {'vmin': -0.5, 'vmax': 0.5, 'cmap': plt.get_cmap("seismic")}
            }

            pcm = ax.imshow(gen_imgs[cnt][:, :, 0], **switcher.get(j, {}))

            ax.axis('off')
            fig.colorbar(pcm, ax=ax)

            cnt += 1

        fig.savefig(f"{subpathID}/imagesAll/{epoch}.png", bbox_inches="tight")
        plt.close()


def sample_images_cycleGAN(epoch, g_AB, g_BA, subpathID, A_test, B_test, img_shape, img_rows, img_cols, images_to_sample, v_min=1, v_max=2, use_boundaries=False):
    os.makedirs(f'{subpathID}/imagesAll', exist_ok=True)
    r, c = len(images_to_sample), 12

    # titles = ['Original', 'Translated [1,2]', 'Translated', 'Reconstructed', 'real - fake', 'real - fake',
    #          'Original', 'Translated [1,2]', 'Translated', 'Reconstructed', 'real - fake', 'real - fake']
    fig, axs = plt.subplots(r, c, figsize=(2.75 * c, 2 * r))

    for i in range(r):
        cnt = 0

        img_A = A_test[images_to_sample[i]].reshape((1, img_rows, img_cols, 1))
        img_B = B_test[images_to_sample[i]].reshape((1, img_rows, img_cols, 1))

        if use_boundaries:
            fake_B = g_AB.predict([img_A, get_boundaries(img_B)])
            fake_A = g_BA.predict([img_B, get_boundaries(img_A)])

            reconstr_A = g_BA.predict([fake_B, get_boundaries(img_A)])
            reconstr_B = g_AB.predict([fake_A, get_boundaries(img_B)])
        else:
            fake_B = g_AB.predict(img_A)
            fake_A = g_BA.predict(img_B)

            reconstr_A = g_BA.predict(fake_B)
            reconstr_B = g_AB.predict(fake_A)

        gen_imgs = np.concatenate(
            [img_A, fake_A, fake_A, reconstr_A, img_A - fake_A, img_A - fake_A,
             img_B, fake_B, fake_B, reconstr_B, img_B - fake_B, img_B - fake_B])

        for j in range(c):
            ax = axs[i, j]

            switcher = {
                1: {'vmin': np.min(img_A), 'vmax': np.max(img_A)},
                4: {'vmin': -0.05, 'vmax': 0.05, 'cmap': plt.get_cmap("seismic")},
                5: {'vmin': -0.5, 'vmax': 0.5, 'cmap': plt.get_cmap("seismic")},
                6: {'vmin': v_min, 'vmax': v_max},
                7: {'vmin': v_min, 'vmax': v_max},
                10: {'vmin': -0.05, 'vmax': 0.05, 'cmap': plt.get_cmap("seismic")},
                11: {'vmin': -0.5, 'vmax': 0.5, 'cmap': plt.get_cmap("seismic")}
            }

            pcm = ax.imshow(gen_imgs[cnt][:, :, 0], **switcher.get(j, {}))

            ax.axis('off')
            fig.colorbar(pcm, ax=ax)

            cnt += 1

        fig.savefig(f"{subpathID}/imagesAll/{epoch}.png", bbox_inches="tight")
        plt.close()


def rewrite_bases(img_rows=64, img_cols=64, numDataZ=6000, num_bases=-1):
    pathData = f"MLdata/{img_rows}x{img_cols}/k1_{img_rows}"

    xAllData = np.load(f"{pathData}_{numDataZ}_all_u.npy")

    num_c1, num_c2, num_rows, num_cols = xAllData.shape
    num_rows -= 2
    num_cols -= 2

    if num_bases == -1:
        num_bases = num_c2

    xBdy = np.zeros((num_c1, num_bases, 4, max(num_rows, num_cols)))

    xBdy[:, :, 0] = xAllData[:, :num_bases, 0, 1:-1] - xAllData[:, :num_bases, 1, 1:-1]  # oben
    xBdy[:, :, 1] += xAllData[:, :num_bases, 1:-1, -1] - xAllData[:, :num_bases, 1:-1, -2]  # rechts
    xBdy[:, :, 2] += xAllData[:, :num_bases, -1, 1:-1] - xAllData[:, :num_bases, -2, 1:-1]  # unten
    xBdy[:, :, 3] += xAllData[:, :num_bases, 1:-1, 0] - xAllData[:, :num_bases, 1:-1, 1]  # links

    np.save(f"{pathData}_{numDataZ}_all_u_compact.npy", xBdy)


def load_data_bases_borders(pathData, numDataZ, numDataTrain, numDataValid, numDataTest, num_bases, shuffleData):
    xAllData = np.load(f"{pathData}_{numDataZ}_all_u_compact.npy")

    num_c1, num_c2, num_borders, num_rows = xAllData.shape
    num_cols = num_rows

    xBdy = np.zeros((num_c1, num_bases, num_rows, num_cols))

    xBdy[:, :, 0, :] = xAllData[:, :num_bases, 0]  # oben
    xBdy[:, :, :, -1] += xAllData[:, :num_bases, 1]  # rechts
    xBdy[:, :, -1, :] += xAllData[:, :num_bases, 2]  # unten
    xBdy[:, :, :, 0] += xAllData[:, :num_bases, 3]  # links

    xBdy[:, :, 0, 0] /= 2
    xBdy[:, :, -1, 0] /= 2
    xBdy[:, :, -1, -1] /= 2
    xBdy[:, :, 0, -1] /= 2

    xBdy = np.moveaxis(xBdy, 1, -1)

    yAllData = np.load(f"{pathData}_{numDataZ}_all_q.npy")
    yAllData = yAllData.reshape((num_c1, num_rows, num_cols, 1))

    if shuffleData:
        randomize = np.arange(len(xAllData))
        np.random.shuffle(randomize)
        xBdy = xBdy[randomize]
        yAllData = yAllData[randomize]

    A_train = xBdy[:numDataTrain]
    A_valid = xBdy[numDataTrain:(numDataTrain + numDataValid)]
    A_test = xBdy[(numDataTrain + numDataValid):(numDataTrain + numDataValid + numDataTest)]

    B_train = yAllData[:numDataTrain]
    B_valid = yAllData[numDataTrain:(numDataTrain + numDataValid)]
    B_test = yAllData[(numDataTrain + numDataValid):(numDataTrain + numDataValid + numDataTest)]

    return [A_train, B_train, A_valid, B_valid, A_test, B_test]


def load_data_bases(pathData, numDataZ, numDataTrain, numDataValid, numDataTest, num_bases, shuffleData):
    xAllData = np.load(f"{pathData}_{numDataZ}_all_u.npy")

    num_c1, num_c2, num_rows, num_cols = xAllData.shape
    num_rows -= 2
    num_cols -= 2

    xBdy = np.zeros((num_c1, num_bases, num_rows, num_cols))

    xBdy[:, :, 0, :] = xAllData[:, :num_bases, 0, 1:-1] - xAllData[:, :num_bases, 1, 1:-1]  # oben
    xBdy[:, :, :, -1] += xAllData[:, :num_bases, 1:-1, -1] - xAllData[:, :num_bases, 1:-1, -2]  # rechts
    xBdy[:, :, -1, :] += xAllData[:, :num_bases, -1, 1:-1] - xAllData[:, :num_bases, -2, 1:-1]  # unten
    xBdy[:, :, :, 0] += xAllData[:, :num_bases, 1:-1, 0] - xAllData[:, :num_bases, 1:-1, 1]  # links

    xBdy[:, :, 0, 0] /= 2
    xBdy[:, :, -1, 0] /= 2
    xBdy[:, :, -1, -1] /= 2
    xBdy[:, :, 0, -1] /= 2

    xBdy = np.moveaxis(xBdy, 1, -1)

    yAllData = np.load(f"{pathData}_{numDataZ}_all_q.npy")
    yAllData = yAllData.reshape((num_c1, num_rows, num_cols, 1))

    if shuffleData:
        randomize = np.arange(len(xAllData))
        np.random.shuffle(randomize)
        xBdy = xBdy[randomize]
        yAllData = yAllData[randomize]

    A_train = xBdy[:numDataTrain]
    A_valid = xBdy[numDataTrain:(numDataTrain + numDataValid)]
    A_test = xBdy[(numDataTrain + numDataValid):(numDataTrain + numDataValid + numDataTest)]

    B_train = yAllData[:numDataTrain]
    B_valid = yAllData[numDataTrain:(numDataTrain + numDataValid)]
    B_test = yAllData[(numDataTrain + numDataValid):(numDataTrain + numDataValid + numDataTest)]

    return [A_train, B_train, A_valid, B_valid, A_test, B_test]


def load_data_bases_block(pathData, numDataZ, numDataTrain, numDataValid, numDataTest, num_bases, shuffleData):
    xAllData = np.load(f"{pathData}_{numDataZ}_all_u.npy")
    yAllData = np.load(f"{pathData}_{numDataZ}_all_q.npy")

    num_c1, num_c2, num_rows, num_cols = xAllData.shape
    num_rows -= 2
    num_cols -= 2

    yAllData = yAllData.reshape((num_c1, num_rows, num_cols, 1))

    xBdy = np.zeros((num_c1, num_bases, 2, 2, num_rows))

    xBdy[:, :, 0, 0, :] = xAllData[:, :num_bases, 0, 1:-1] - xAllData[:, :num_bases, 1, 1:-1]  # oben
    xBdy[:, :, 0, 1, :] += xAllData[:, :num_bases, 1:-1, -1] - xAllData[:, :num_bases, 1:-1, -2]  # rechts
    xBdy[:, :, 1, 1, :] += xAllData[:, :num_bases, -1, 1:-1] - xAllData[:, :num_bases, -2, 1:-1]  # unten
    xBdy[:, :, 1, 0, :] += xAllData[:, :num_bases, 1:-1, 0] - xAllData[:, :num_bases, 1:-1, 1]  # links

    xBdy = np.moveaxis(xBdy, 1, -1)
    xBdy = xBdy.reshape((num_c1, 2, 2, -1))

    if shuffleData:
        randomize = np.arange(len(xAllData))
        np.random.shuffle(randomize)
        # xAllData = xAllData[randomize]
        xBdy = xBdy[randomize]
        yAllData = yAllData[randomize]

    A_train = xBdy[:numDataTrain]
    A_valid = xBdy[numDataTrain:(numDataTrain + numDataValid)]
    A_test = xBdy[(numDataTrain + numDataValid):(numDataTrain + numDataValid + numDataTest)]

    B_train = yAllData[:numDataTrain]
    B_valid = yAllData[numDataTrain:(numDataTrain + numDataValid)]
    B_test = yAllData[(numDataTrain + numDataValid):(numDataTrain + numDataValid + numDataTest)]

    return [A_train, B_train, A_valid, B_valid, A_test, B_test]


def load_batch(numDataTrain, A_train, B_train, img_shape, img_rows, img_cols, batch_size=1):
    n_batches = int(numDataTrain / batch_size)
    total_samples = n_batches * batch_size

    order = np.random.choice(range(total_samples), total_samples, replace=False)

    for i in range(n_batches):
        imgs_A = A_train[order[i * batch_size:(i + 1) * batch_size]].reshape((batch_size,) + img_shape)
        imgs_B = B_train[order[i * batch_size:(i + 1) * batch_size]].reshape((batch_size, img_rows, img_cols, 1))

        yield imgs_A, imgs_B


def train(epochs, net, num_data_train, all_sets, img_shape, img_rows, img_cols, tensorboard, images_to_sample, path, ID, subpathID, netType, versionNr, batch_size=1, sample_interval=50, start_epoch=0, change_after=15, max_bs=256):
    A_train, B_train, A_valid, B_valid, A_test, B_test = all_sets

    best_loss = np.infty

    act_batch_size = batch_size // 2

    for epoch in range(epochs):
        epoch += start_epoch
        e0 = time.time()

        for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(num_data_train, A_train, B_train, img_shape, img_rows, img_cols, batch_size=act_batch_size)):
            net.train_on_batch(imgs_A, imgs_B)

        loss = net.evaluate(x=A_train, y=B_train, verbose=2)
        v_loss = net.evaluate(x=A_valid, y=B_valid, verbose=2)

        e1 = time.time()
        e_diff = e1 - e0

        print(f"{ID} {epoch}: {loss} {v_loss} {loss - v_loss} {time.ctime(e1)} -> {time.ctime(e1 + e_diff)} ; {act_batch_size}")

        write_log(tensorboard, ['loss'], [loss], epoch)
        write_log(tensorboard, ['val_loss'], [v_loss], epoch)
        write_log(tensorboard, ['zzz_diff'], [loss - v_loss], epoch)

        if v_loss < best_loss:
            best_loss = v_loss

            net.save(f"{path}{subpathID}/{ID}_comb_{netType}{versionNr}_best.h5")

        if epoch % sample_interval == 0 and epoch > 0:
            net.save(os.path.join(path, f"{path}{subpathID}/{ID}_comb_{netType}{versionNr}.h5"))
            sample_images(epoch, net, subpathID, A_test, B_test, img_shape, img_rows, img_cols, images_to_sample)

        if epoch % change_after == 0 and act_batch_size < max_bs:
            act_batch_size *= 2

    ##################


def conc(data):
    if len(data) == 1:
        ret_val = data[0]
    else:
        temp = []
        for part in data:
            if not part == []:
                temp.append(part)

        if len(temp) == 1:
            ret_val = temp[0]
        else:
            ret_val = Concatenate()(temp)

    return ret_val


def conv(layer_input, filters, k_size, nStrides=1, k_init="glorot_uniform", pad="same", parts=0, dil_max=-1, activation=Mish, act_params=None, refl_padding=True):
    if act_params is None:
        act_params = {}

    d = layer_input
    striding1 = (1, 1)
    striding2 = (1, 1)
    pad_size = (k_size - 1) // 2

    if refl_padding: pad = "valid"

    for i in range(parts + 1):
        if i == parts:
            striding1 = (nStrides, 1)
            striding2 = (1, nStrides)

        if refl_padding: d = ReflectionPadding2D(padding=(pad_size, pad_size))(d)
        d = Conv2D(filters, kernel_size=(k_size, 1), strides=striding1, padding=pad, kernel_initializer=k_init)(d)
        d = Conv2D(filters, kernel_size=(1, k_size), strides=striding2, padding=pad, kernel_initializer=k_init)(d)
        d = activation(**act_params)(d)

    return d


def conv_act(layer_input, filters, k_size, nStrides=1, k_init="glorot_uniform", pad="same", parts=0, activation=Mish, act_params=None, refl_padding=True):
    if act_params is None:
        act_params = {}

    result = layer_input
    striding1 = (1, 1)
    striding2 = (1, 1)
    pad_size = (k_size - 1) // 2

    if refl_padding: pad = "valid"

    for i in range(parts + 1):
        d = result
        if i == parts:
            striding1 = (nStrides, 1)
            striding2 = (1, nStrides)

        if refl_padding: d = ReflectionPadding2D(padding=(pad_size, 1))(d)
        d = Conv2D(filters, kernel_size=(k_size, 1), strides=striding1, padding=pad, kernel_initializer=k_init)(d)
        d = activation(**act_params)(d)
        if refl_padding: d = ReflectionPadding2D(padding=(1, pad_size))(d)
        d = Conv2D(filters, kernel_size=(1, k_size), strides=striding2, padding=pad, kernel_initializer=k_init)(d)
        d = activation(**act_params)(d)
        result = conc([d, result])

    return result


def conv_both(layer_input, filters, k_size, nStrides=1, k_init="glorot_uniform", pad="same", parts=0, activation=Mish, act_params=None, refl_padding=True):
    if act_params is None:
        act_params = {}

    result = layer_input

    striding1 = (1, 1)
    striding2 = (1, 1)
    pad_size = (k_size - 1) // 2

    if refl_padding: pad = "valid"

    for i in range(parts + 1):
        d1 = result
        d2 = result

        if i == parts:
            striding1 = (nStrides, 1)
            striding2 = (1, nStrides)

        if refl_padding: d1 = ReflectionPadding2D(padding=(pad_size, 1))(result)
        d1 = Conv2D(filters, kernel_size=(k_size, 1), strides=striding1, padding=pad, kernel_initializer=k_init)(d1)
        if refl_padding: d1 = ReflectionPadding2D(padding=(1, pad_size))(d1)
        d1 = Conv2D(filters, kernel_size=(1, k_size), strides=striding2, padding=pad, kernel_initializer=k_init)(d1)
        d1 = activation(**act_params)(d1)

        if refl_padding: d2 = ReflectionPadding2D(padding=(1, pad_size))(result)
        d2 = Conv2D(filters, kernel_size=(1, k_size), strides=striding2, padding=pad, kernel_initializer=k_init)(d2)
        if refl_padding: d2 = ReflectionPadding2D(padding=(pad_size, 1))(d2)
        d2 = Conv2D(filters, kernel_size=(k_size, 1), strides=striding1, padding=pad, kernel_initializer=k_init)(d2)
        d2 = activation(**act_params)(d2)

        result = conc([d1, d2, result])

    return result


def conv_both_act(layer_input, filters, k_size, nStrides=1, k_init="glorot_uniform", pad="same", parts=0, dil_max=-1, activation=Mish, act_params=None, refl_padding=True):
    if act_params is None:
        act_params = {}

    result = layer_input
    striding1 = (1, 1)
    striding2 = (1, 1)
    pad_size = (k_size - 1) // 2

    if refl_padding: pad = "valid"

    for i in range(parts + 1):
        if i == parts:
            striding1 = (nStrides, 1)
            striding2 = (1, nStrides)

        if refl_padding: d1 = ReflectionPadding2D(padding=(pad_size, 1))(result)
        d1 = Conv2D(filters, kernel_size=(k_size, 1), strides=striding1, padding=pad, kernel_initializer=k_init)(d1)
        d1 = activation(**act_params)(d1)
        if refl_padding: d1 = ReflectionPadding2D(padding=(1, pad_size))(d1)
        d1 = Conv2D(filters, kernel_size=(1, k_size), strides=striding2, padding=pad, kernel_initializer=k_init)(d1)
        d1 = activation(**act_params)(d1)

        if refl_padding: d2 = ReflectionPadding2D(padding=(1, pad_size))(result)
        d2 = Conv2D(filters, kernel_size=(1, k_size), strides=striding2, padding=pad, kernel_initializer=k_init)(d2)
        d2 = activation(**act_params)(d2)
        if refl_padding: d2 = ReflectionPadding2D(padding=(pad_size, 1))(d2)
        d2 = Conv2D(filters, kernel_size=(k_size, 1), strides=striding1, padding=pad, kernel_initializer=k_init)(d2)
        d2 = activation(**act_params)(d2)

        result = conc([d1, d2, result])

    return result


def convComb(layer_input, filters, k_size, nStrides=1, k_init="glorot_uniform", pad="same", parts=0, dil_max=1, activation=Mish, act_params=None):
    if act_params is None:
        act_params = {}

    result = layer_input
    filters_part = max(1, filters // dil_max)

    for _ in range(parts):
        d = result
        d = Conv2D(filters, kernel_size=k_size, padding=pad, kernel_initializer=k_init)(d)
        d = activation(**act_params)(d)

        result = conc([result, d])

    d_l = result
    if dil_max > 0: result = []

    for i in range(dil_max):
        d = Conv2D(filters_part, kernel_size=k_size, dilation_rate=i + 1, padding=pad, kernel_initializer=k_init)(d_l)
        d = activation(**act_params)(d)
        d = Conv2D(filters_part, kernel_size=k_size, strides=nStrides, padding=pad, kernel_initializer=k_init)(d)
        d = activation(**act_params)(d)

        result = conc([result, d])

    return result


def convT(layer_input, filters, k_size, nStrides, k_init="glorot_uniform", pad="same", parts=0, skip=None, activation=Mish, act_params=None, split_kernel=True):
    if act_params is None:
        act_params = {}

    d = layer_input

    if split_kernel:
        d = Conv2DTranspose(filters, kernel_size=(k_size - 1, 1), strides=(nStrides, 1), padding='same', kernel_initializer=k_init)(d)
        d = Conv2DTranspose(filters, kernel_size=(1, k_size - 1), strides=(1, nStrides), padding='same', kernel_initializer=k_init)(d)
        d = activation(**act_params)(d)
    else:
        d = Conv2DTranspose(filters, kernel_size=k_size - 1, strides=nStrides, padding='same', kernel_initializer=k_init)(d)
        d = activation(**act_params)(d)

    if not skip is None:
        d = conc([d, skip])

    return repetition(d, filters, k_size, pad, k_init, parts, activation=activation, act_params=act_params) if parts > 0 else d


def repetition(layer_input, filters, k_size, pad, k_init, parts, pad_size=0, activation=Mish, act_params=None):
    if act_params is None:
        act_params = {}

    d = layer_input
    result = d

    for _ in range(parts):
        d = ReflectionPadding2D(padding=(pad_size, pad_size))(d)
        d = conv(d, filters, k_size, k_init=k_init, pad=pad, parts=parts, activation=activation, act_params=act_params)
        d = activation(**act_params)(d)

        result = conc([result, d])

    return result


def convUp(layer_input, filters, k_size, nStrides=1, k_init="he_uniform", pad="valid", parts=0, skip=None, activation=Mish, act_params=None):
    if act_params is None:
        act_params = {}

    k_init = "glorot_uniform"
    pad_size = int((k_size - 1) / 2)

    d1 = UpSampling2D(nStrides, interpolation='bilinear')(layer_input)
    d2 = UpSampling2D(nStrides, interpolation='nearest')(layer_input)

    d = conc([d1, d2])
    result = d

    for _ in range(parts + 1):
        d = conv(d, filters, k_size, k_init=k_init, pad=pad, parts=parts, activation=activation, act_params=act_params)
        d = activation(**act_params)(d)

        result = conc([result, d])

    if not skip is None:
        result = conc([result, skip])

    return result


def Crop(dim, start, end, step=1, **kwargs):
    def func(x):
        dimension = dim
        if dimension == -1:
            dimension = len(x.shape) - 1
        if dimension == 0:
            return x[start:end:step]
        if dimension == 1:
            return x[:, start:end:step]
        if dimension == 2:
            return x[:, :, start:end:step]
        if dimension == 3:
            return x[:, :, :, start:end:step]
        if dimension == 4:
            return x[:, :, :, :, start:end:step]

    return Lambda(func, **kwargs)


def split_block(d0, upConv, factor=1, k_size=1, stride=1, num_parts=0):
    act_shape = int(d0.get_shape()[-1])

    temp = 16

    dimension = act_shape // temp if act_shape > temp else 1
    mult = act_shape // dimension

    branch_outputs = []

    ###############
    use_overlap = True
    num_parts1 = 0

    if use_overlap:
        mult_h = mult // 2

        if stride is 1:
            branch_outputs.append(conv(Crop(3, 0, mult_h // 2 + 1)(d0), 1 * factor, 3, 1, parts=num_parts1))
        else:
            branch_outputs.append(upConv(Crop(3, 0, mult_h // 2 + 1)(d0), 1 * factor, 3, 2, parts=num_parts1))

        for i in range(2 * dimension - 1):
            out = Crop(3, i * mult_h, (i + 1) * mult_h)(d0)
            # branch_outputs.append(conv(out, 1 * factor, 3, 1, parts=num_parts1))
            if stride is 1:
                branch_outputs.append(conv(out, factor, k_size, stride, parts=num_parts))
            else:
                branch_outputs.append(upConv(out, factor, k_size, stride, parts=num_parts))
    else:
        for i in range(dimension):
            out = Crop(3, i * mult, (i + 1) * mult)(d0)
            if stride is 1:
                branch_outputs.append(conv(out, factor, k_size, stride, parts=num_parts))
            else:
                branch_outputs.append(upConv(out, factor, k_size, stride, parts=num_parts))

    ###################

    # for i in range(dimension):
    #     out = Crop(3, i * mult, (i + 1) * mult)(d0)
    #
    #     if stride is 1:
    #         branch_outputs.append(conv(out, factor, k_size, stride, parts=num_parts))
    #     else:
    #         branch_outputs.append(upConv(out, factor, k_size, stride, parts=num_parts))

    return branch_outputs.pop() if dimension is 1 else conc(branch_outputs)
