from __future__ import print_function, division
import scipy
import tensorflow as tf
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import librosa

class CycleSpecGAN():
    def __init__(self):
        self.source_data = np.load("source.npz")
        self.target_data = np.load("target.npz")
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'IRMAS-pia-2-gac'
        self.model_name = 'cyclespecgan-no-identity'
        self.timestamp = datetime.datetime.now()


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0      # Cycle-consistency loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_A.name = "d_A"

        self.d_A.compile(loss='mse',
        optimizer=optimizer,
        metrics=['accuracy'])

        self.d_B = self.build_discriminator()
        self.d_B.name = "d_B"

        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_AB.name = "g_AB"

        self.g_BA = self.build_generator()
        self.g_BA.name = "g_BA"

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                        reconstr_A, reconstr_B ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',],
                            metrics=['accuracy'],
                            loss_weights=[ 1, 1, self.lambda_cycle, self.lambda_cycle ],
                            optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def load_batch(self, source_specs, target_specs, batch_size):
        n_batches = int(min(source_specs.shape[0], target_specs.shape[0]) / batch_size)
        total_samples = n_batches * batch_size
        source_idxs = np.random.choice(source_specs.shape[0], total_samples, replace=False)
        target_idxs = np.random.choice(target_specs.shape[0], total_samples, replace=False)

        for i in range(n_batches-1):
            batch_A = source_idxs[i*batch_size:(i+1)*batch_size]
            batch_B = target_idxs[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            
            for idx_A, idx_B in zip(batch_A, batch_B):
                img_A = source_specs[idx_A, :, :]
                img_B = source_specs[idx_B, :, :]
                
                imgs_A.append(img_A)
                imgs_B.append(img_B)


            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)
            yield imgs_A, imgs_B, n_batches

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        self.checkpoint_dir = 'checkpoints/%s-%s' % (self.dataset_name, self.timestamp)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Writer for tensorboard logs
        writer = tf.summary.FileWriter("./logs/cyclespecgan_{}".format(datetime.datetime.now()))

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        source_specs = self.source_data["specs"]
        source_specs = source_specs[:, :, :, None]

        target_specs = self.target_data["specs"]
        target_specs = target_specs[:, :, :, None]

        # for denormalizing mel_spectrogram
        source_mel_means = self.source_data["mean"]
        source_mel_stds = self.source_data["std"]

        target_mel_means = self.target_data ["mean"]
        target_mel_stds = self.target_data ["std"]

        for epoch, _ in enumerate(range(epochs), start=1):
            for batch_i, (imgs_A, imgs_B, n_batches) in enumerate(self.load_batch(source_specs, target_specs, batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                summary_values = [
                    tf.Summary.Value(tag="D_loss", simple_value=d_loss[0]),
                    tf.Summary.Value(tag="D_acc", simple_value=d_loss[1]),
                    tf.Summary.Value(tag="G_loss", simple_value=g_loss[0]),
                    tf.Summary.Value(tag="G_adv", simple_value=np.mean(g_loss[1:3])),
                    tf.Summary.Value(tag="G_recon", simple_value=np.mean(g_loss[3:5])),
                    tf.Summary.Value(tag="G_AB_acc", simple_value=g_loss[8]),
                    tf.Summary.Value(tag="G_BA_acc", simple_value=g_loss[7]),
                ]
                summary = tf.Summary(value=summary_values)
                writer.add_summary(summary)
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %f] [G loss: %05f, adv: %05f, recon: %05f, g_AB_acc: %f, g_BA_acc: %f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, n_batches,
                                                                            d_loss[0], d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            g_loss[8],
                                                                            g_loss[7],
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
            
            # Save models per epoch
            self.g_AB.save("%s/g_AB-epoch-%s.h5" % (self.checkpoint_dir, epoch))
            self.g_BA.save("%s/g_BA-epoch-%s.h5" % (self.checkpoint_dir, epoch))
            # self.d_A.save("%s/d_A-epoch-%s.h5" % (self.checkpoint_dir, epoch))
            # self.d_B.save("%s/d_B-epoch-%s.h5" % (self.checkpoint_dir, epoch))
            # self.combined.save("%s/combined-epoch-%s.h5" % (self.checkpoint_dir, epoch))

    def sample_images(self, epoch, batch_i):
        output_path = 'outputs/%s/%s' % (self.dataset_name, self.timestamp)
        os.makedirs(output_path, exist_ok=True)
        r, c = 2, 3

        source_specs = self.source_data["specs"]
        source_specs = source_specs[:, :, :, None]

        target_specs = self.target_data["specs"]
        target_specs = target_specs[:, :, :, None]

        img_A = source_specs[0]
        img_B = target_specs[0]

        imgs_A = np.array([img_A])
        imgs_B = np.array([img_B])

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.array([
            self.to_img(imgs_A[0]),
            self.to_img(fake_B[0]),
            self.to_img(reconstr_A[0]),
            self.to_img(imgs_B[0]),
            self.to_img(fake_A[0]),
            self.to_img(reconstr_B[0])
        ])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = [
            ['Original-A', 'Translated-A2B', 'Reconstructed-A'],
            ['Original-B', 'Translated-B2A', 'Reconstructed-B']
        ]
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt], cmap='gray', vmin=0, vmax=255)
                axs[i, j].set_title(titles[i][j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig('%s/epoch-%s-batch-%s.png' % (output_path, epoch, batch_i))
        plt.close()

    def to_img(self, mat):
        img = (mat * 127.5) + 127.5
        img = np.squeeze(np.round(img)).astype(np.uint8)
        return img