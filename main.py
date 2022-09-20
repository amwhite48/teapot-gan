import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import teapot_data.teapot_dataset.teapot_dataset

# via keras https://keras.io/examples/generative/wgan_gp/

train_data, test_data = tfds.load('teapot_dataset', split=['train[:80%]', 'train[80%:]'])


leaky_relu = tf.keras.layers.LeakyReLU(0.01)
relu = tf.keras.layers.Activation(activation='relu')
batch_norm = tf.keras.layers.BatchNormalization()

def get_dis_model(name="dis_model"):
    return tf.keras.Sequential([
        tf.keras.layers.Conv3D(128, (4,4,4), 2, padding='same'),
        batch_norm,
        leaky_relu,
        tf.keras.layers.Conv3D(256, (4,4,4), 2, padding='same'),
        batch_norm,
        leaky_relu,
        tf.keras.layers.Conv3D(1, (4,4,4), 2, padding='valid'),
        batch_norm,
        leaky_relu,
        tf.keras.layers.Activation(activation='sigmoid')
    ], name=name)


def get_gen_model(name="gen_model"):
    return tf.keras.Sequential([
        tf.keras.layers.Conv3DTranspose(128, (4,4,4), 1, padding='valid'),
        batch_norm,
        relu,
        tf.keras.layers.Conv3DTranspose(32, (4,4,4), 2, padding='same'),
        batch_norm,
        relu,
        tf.keras.layers.Conv3DTranspose(1, (4,4,4), 2, padding='same'),
        batch_norm,
        relu,
        tf.keras.layers.Activation(activation='sigmoid')
    ], name=name)

class GAN_Core(tf.keras.Model):

    '''
    Core class to administrate both the generator and discriminator
    Code Struture via Brown Deep Learning Lab 8 Spring 2022
    '''

    def __init__(self, dis_model, gen_model, z_dims, z_sampler=tf.random.normal, **kwargs):
        '''
        self.gen_model = generator model;           z_like -> x_like
        self.dis_model = discriminator model;       x_like -> probability
        self.z_sampler = sampling strategy for z;   z_dims -> z
        self.z_dims    = dimensionality of generator input
        '''
        super().__init__(**kwargs)
        self.z_dims = z_dims
        self.z_sampler = z_sampler
        self.gen_model = gen_model
        self.dis_model = dis_model

    def sample_z(self, num_samples, **kwargs):
        return self.z_sampler([num_samples, *self.z_dims[1:]])
    
    def discriminate(self, inputs, **kwargs):
        return self.dis_model(inputs, **kwargs)

    def generate(self, z, **kwargs):
        return self.gen_model(z, **kwargs)

    
    def call(self, inputs, **kwargs):
        b_size = tf.shape(inputs)[0]
        z_samp = self.sample_z(b_size)  ## Generate a z sample
        g_samp = self.generate(z_samp)   ## Generate an x-like image
        d_samp = self.discriminate(g_samp)   ## Predict whether x-like is real
        print(f'Z( ) Shape = {z_samp.shape}')
        print(f'G(z) Shape = {g_samp.shape}')
        print(f'D(x) Shape = {d_samp.shape}\n')
        return d_samp

    def build(self, **kwargs):
        super().build(input_shape=self.z_dims, **kwargs)

gan = GAN_Core(    
    dis_model = get_dis_model(), 
    gen_model = get_gen_model(), 
    z_dims = (1, 1, 1, 1, 128),
    name="gan"
)

gan.build()
gan.summary()