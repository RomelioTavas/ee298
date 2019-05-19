import numpy as np

import datetime
import tensorflow as tf

from keras.layers import (Conv1D, Dense, Dropout, Input, Concatenate,
                          LeakyReLU, Flatten, UpSampling1D)
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
from keras.models import Model

np.random.seed(1001)


class CycleGAN():

    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        self.train_steps = 40000
        self.latent_size = self.config.audio_length
        self.model_name = 'cyclegan_audio'
        self.batch_size = 8

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

    def build_model(self):
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        print("d_A: ")
        self.d_A.summary()

        self.d_B.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        print("d_B: ")
        self.d_B.summary()

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        print("g_AB: ")
        self.g_AB.summary()
        
        self.g_BA = self.build_generator()
        print("g_BA: ")
        self.g_BA.summary()

        print("Constructing Adversarial network")
        # Input audio from both domains
        audio_A = Input(shape=(self.config.audio_length, 1))
        audio_B = Input(shape=(self.config.audio_length, 1))

        # Translate audios to the other domain
        fake_B = self.g_AB(audio_A)
        fake_A = self.g_BA(audio_B)
        # Translate audios back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of audios
        audio_A_id = self.g_BA(audio_A)
        audio_B_id = self.g_AB(audio_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated audios
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[audio_A, audio_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       audio_A_id, audio_B_id])
        print("Compiling Adversarial Network")
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)
        print("CycleGAN: ")
        self.combined.summary()

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv1D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        audio = Input(shape=(self.config.audio_length, 1))

        d1 = d_layer(audio, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        
        x = Flatten()(d4)
        out = Dense(1)(x)

        return Model(audio, out)

    def build_generator(self):
        """U-Net Generator"""

        def conv1d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv1D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv1d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling1D(size=2)(layer_input)
            u = Conv1D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Audio input
        d0 = Input(shape=(self.config.audio_length, 1))

        # Downsampling
        d1 = conv1d(d0, self.gf)
        d2 = conv1d(d1, self.gf*2)
        d3 = conv1d(d2, self.gf*4)
        d4 = conv1d(d3, self.gf*8)

        # Upsampling
        u1 = deconv1d(d4, d3, self.gf*4)
        u2 = deconv1d(u1, d2, self.gf*2)
        u3 = deconv1d(u2, d1, self.gf)

        u4 = UpSampling1D(size=2)(u3)
        output_audio = Conv1D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_audio)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        writer = tf.summary.FileWriter("./logs/cyclegan_{}".format(datetime.datetime.now()))
        # Adversarial loss ground truths
        valid = np.ones((batch_size,))
        fake = np.zeros((batch_size,))
        
        for epoch in range(epochs):
            for batch_i, (audios_A, audios_B) in enumerate(self.data_loader.load_batch(batch_size)):
                # ----------------------
                #  Train Discriminators
                # ----------------------
                # Translate images to opposite domain
                fake_B = self.g_AB.predict(audios_A)
                fake_A = self.g_BA.predict(audios_B)

                # Train the discriminators (original audio = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(audios_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(audios_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                # print("Training combined")
                g_loss = self.combined.train_on_batch([audios_A, audios_B],
                                                      [valid, valid,
                                                       audios_A, audios_B,
                                                       audios_A, audios_B])

                # # Plot the progress
                elapsed_time = datetime.datetime.now() - start_time
                summary_values = [
                    tf.Summary.Value(tag="D loss", simple_value=d_loss[0]),
                    tf.Summary.Value(tag="D acc", simple_value=100*d_loss[1]),
                    tf.Summary.Value(tag="G loss", simple_value=g_loss[0]),
                    tf.Summary.Value(tag="G adv", simple_value=np.mean(g_loss[1:3])),
                    tf.Summary.Value(tag="G recon", simple_value=np.mean(g_loss[3:5])),
                    tf.Summary.Value(tag="G id", simple_value=np.mean(g_loss[5:6]))
                ]
                summary = tf.Summary(value=summary_values)
                writer.add_summary(summary)
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s "
                      % (epoch + 1, epochs,
                          batch_i, self.data_loader.n_batches,
                          d_loss[0], 100*d_loss[1],
                          g_loss[0],
                          np.mean(g_loss[1:3]),
                          np.mean(g_loss[3:5]),
                          np.mean(g_loss[5:6]),
                          elapsed_time))

                # If at save interval => save generated audio samples
                # if batch_i % sample_interval == 0:
                #     self.sample_audio(epoch, batch_i)
            self.g_AB.save("cyclegan_A2B_epoch_{}.h5".format(epoch))
            self.g_BA.save("cyclegan_B2A_epoch_{}.h5".format(epoch))