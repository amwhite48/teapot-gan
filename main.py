import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import teapot_data.teapot_dataset.teapot_dataset


ds = list(tfds.load('teapot_dataset')['train'])
train_data = np.asarray(list(map(lambda x: x['voxels'], ds)))
print(type(train_data))

leaky_relu = tf.keras.layers.LeakyReLU(0.01)
relu = tf.keras.layers.Activation(activation='relu')
batch_norm = tf.keras.layers.BatchNormalization()
latent_dim = 128
batch_size = 100

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
        return d_samp

    def build(self, **kwargs):
        super().build(input_shape=self.z_dims, **kwargs)
    
    def compile(self, optimizers, losses, accuracies, **kwargs):
        super().compile(
            loss        = losses.values(),
            optimizer   = optimizers.values(),
            metrics     = accuracies.values(),
            **kwargs
        )
        self.loss_funcs = losses
        self.optimizers = optimizers
        self.acc_funcs  = accuracies

    def fit(self, *args, d_steps=1, g_steps=1, **kwargs):
        self.g_steps = g_steps
        self.d_steps = d_steps
        super().fit(*args, **kwargs)


    def test_step(self, data): 
        x_real = data
        batch_size = tf.shape(x_real)[0]

        x_fake = self.generate(self.sample_z(batch_size))
        d_fake = self.discriminate(x_fake)
        d_real = self.discriminate(tf.expand_dims(x_real, -1))

        all_funcs = {**self.loss_funcs, **self.acc_funcs}
        return { key : fun(d_fake, d_real) for key, fun in all_funcs.items() }

    def train_step(self, data):
        x_real = data
        batch_size = x_real.shape[0]

        sample = self.sample_z(batch_size)
          
        loss_fn   = self.loss_funcs['d_loss']
        optimizer = self.optimizers['d_opt']
        for i in range(self.d_steps):
          with tf.GradientTape() as tape:
            x_fake = self.generate(sample)
            d_fake = self.discriminate(x_fake)
            d_real = self.discriminate(tf.expand_dims(x_real, -1))
            loss = loss_fn(d_fake, d_real)
          gradients = tape.gradient(loss, self.dis_model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, self.dis_model.trainable_variables))

        loss_fn   = self.loss_funcs['g_loss']
        optimizer = self.optimizers['g_opt'] 

        for i in range(self.g_steps):
          with tf.GradientTape() as tape:
            x_fake = self.generate(sample)
            d_fake = self.discriminate(x_fake)
            d_real = self.discriminate(tf.expand_dims(x_real, -1))
            loss = loss_fn(d_fake, d_real)
          gradients = tape.gradient(loss, self.gen_model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, self.gen_model.trainable_variables))

        all_funcs = {**self.loss_funcs, **self.acc_funcs}
        return { key : fun(d_fake, d_real) for key, fun in all_funcs.items() }


gan = GAN_Core(    
    dis_model = get_dis_model(), 
    gen_model = get_gen_model(), 
    z_dims = (1, 1, 1, 1, latent_dim),
    name="gan"
)

class EpochVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, model, sample_inputs, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.sample_inputs = sample_inputs
        self.imgs = [] 

    def on_epoch_end(self, epoch, logs=None):
        x_real, z_samp = self.sample_inputs
        x_fake = self.model.gen_model(z_samp)
        # d_real = tf.nn.sigmoid(self.model.dis_model(x_real))
        # d_fake = tf.nn.sigmoid(self.model.dis_model(x_fake))
        for i in range(x_fake.shape[0]):
            filename = "outputs/teapot_out_e" + str(epoch) + "_" + str(i) + ".npy"
            with open(filename, 'wb') as f:
                np.save(filename, np.asarray(np.asarray(x_fake[i])))


gan.build()
gan.dis_model.summary()
gan.gen_model.summary()

acc_func = tf.keras.metrics.binary_accuracy     

def g_acc(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(acc_func(tf.ones_like(d_fake), d_fake))

def d_acc_fake(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    return tf.reduce_mean(acc_func(tf.zeros_like(d_fake), d_fake))

def d_acc_real(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    return tf.reduce_mean(acc_func(tf.ones_like(d_real), d_real))


gan.compile(
    optimizers = {
        'd_opt' : tf.keras.optimizers.Adam(1e-3, beta_1=0.5), 
        'g_opt' : tf.keras.optimizers.Adam(1e-3, beta_1=0.5), 
    },
    losses = {
        'd_loss' : tf.keras.losses.BinaryCrossentropy(),
        'g_loss' : tf.keras.losses.BinaryCrossentropy(),
    },
    accuracies = {
        'd_acc_real' : d_acc_real,
        'd_acc_fake' : d_acc_fake,
        'g_acc'      : g_acc,
    }
)

train_num = 10000       ## Feel free to bump this up to 50000 when your architecture is done
true_sample = train_data[train_num-2:train_num+2]       ## 4 real images
fake_sample = gan.sample_z(4)             ## 4 z realizations
viz_callback = EpochVisualizer(gan, [true_sample, fake_sample])

gan.fit(
    train_data[:train_num], 
    d_steps    = 5, 
    g_steps    = 5, 
    epochs     = 10, ## Feel free to bump this up to 20 when your architecture is done
    batch_size = 50,
    callbacks  = [viz_callback]
)
