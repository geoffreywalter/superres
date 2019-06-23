import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Input, Activation, Lambda, BatchNormalization, Add
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, LambdaCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback
from helpfunc import PS, perceptual_distance
from models import SRResNet, generator, discriminator


#GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 50
config.batch_size = 32
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256

val_dir = 'data/test'
train_dir = 'data/train'

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size

class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        in_sample_images, out_sample_images = next(image_generator(config.batch_size, val_dir))
        preds = self.model.predict(in_sample_images)
        in_resized = []
        # Simple upsampling
        for arr in in_sample_images:
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        test_img = np.zeros(
            (config.output_width, config.output_height, 3))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)

def image_generator(batch_size, img_dir, augment=False):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        random.shuffle(input_filenames)
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0

        if (img_dir == train_dir and augment):
            #Data augmentation
            data_gen_args = dict(#featurewise_center=True,
                        #featurewise_std_normalization=True,
                        #zca_whitening=True,
                        #rotation_range=90,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        vertical_flip=True,
                        #shear_range=0.2,
                        zoom_range=0.2)
            image_datagen = ImageDataGenerator(**data_gen_args)

            seed = counter
            image_datagen.fit(small_images, augment=True, seed=seed)
            gen1 = image_datagen.flow(small_images, batch_size=config.batch_size, shuffle=False, seed=seed)
            gen2 = image_datagen.flow(large_images, batch_size=config.batch_size, shuffle=False, seed=seed)
            gen = zip(gen1, gen2)

            small_images_augmented, large_images_augmented = next(gen)
            yield (small_images_augmented, large_images_augmented)
        else:
            yield (small_images, large_images)
        counter += batch_size


# Neural network
#input1 = Input(shape=(config.input_height, config.input_width, 3), dtype='float32')

#model = Model(inputs=input1, outputs=SRResNet(input1))

config.adversarial_epochs = 1000
config.discriminator_epochs = 1
config.discriminator_examples = 10000
config.generator_epochs = 12
config.generator_examples = 10000
config.generator_seed_dim = 10
config.generator_conv_size = 64
config.batch_size = 100
config.image_shape = (28, 28, 1)

def log_generator(epoch, logs):
    wandb.log({'generator_loss': logs['loss'],
                     'generator_acc': logs['acc'],
                     'discriminator_loss': 0.0,
                     'discriminator_acc': (1-logs['acc'])/2.0+0.5})
def log_discriminator(epoch, logs):
    wandb.log({
            'generator_loss': 0.0,
            'generator_acc': (1.0-logs['acc'])*2.0,
            'discriminator_loss': logs['loss'],
            'discriminator_acc': logs['acc']})

def mix_data(data, generator, length=1000):
    num_examples=int(length/2)

    data= data[:num_examples, :, :]

    seeds = np.random.normal(0, 1, (num_examples, config.generator_seed_dim))

    fake_train = generator.predict(seeds)[:,:,:,0]

    combined  = np.concatenate([ data, fake_train ])

    # combine them together
    labels = np.zeros(combined.shape[0])
    labels[:data.shape[0]] = 1

    indices = np.arange(combined.shape[0])
    np.random.shuffle(indices)
    combined = combined[indices]
    labels = labels[indices]
    combined.shape += (1,)

    labels = np_utils.to_categorical(labels)

    add_noise(labels)

    return (combined, labels)

            
def train_discriminator(generator, discriminator, x_train, x_test, iter):
    train_ilr, train_ihr = mix_data(x_train, image_generator(config.batch_size, train_dir), config.generator_examples)
    test_ilr, test_ihr   = mix_data(x_test, image_generator(config.batch_size, val_dir), config.generator_examples)
    
    discriminator.trainable = True
    discriminator.summary()
    wandb_logging_callback = LambdaCallback(on_epoch_end=log_discriminator)
    
    history = discriminator.fit(train_ilr, train_ihr,
        epochs=config.discriminator_epochs,
        batch_size=config.batch_size, validation_data=(test_ilr, test_ihr),
        callbacks = [wandb_logging_callback])

    discriminator.save(path.join(wandb.run.dir, "discriminator.h5"))

def train_generator(generator, discriminator, joint_model):
    num_examples = config.generator_examples
    train_ilr, train_ihr = next(image_generator(config.batch_size, train_dir))

    discriminator.trainable = False
    
    wandb_logging_callback = LambdaCallback(on_epoch_end=log_generator)
    joint_model.summary()

    joint_model.fit(train_ilr, train_ihr, epochs=config.generator_epochs,
            batch_size=config.batch_size,
            callbacks=[wandb_logging_callback])

    generator.save(path.join(wandb.run.dir, "generator.h5"))
    
    
def train():
    discriminator = create_discriminator()
    generator     = create_generator()    
    joint_model = create_gan(generator, discriminator)
    
    for i in range(config.adversarial_epochs):
        train_discriminator(generator, discriminator, x_train, x_test, i)
        train_generator(generator, discriminator, joint_model)
        #sample_images(generator)

train()

    # for epoch in range(config.num_epochs):
        ##Train discriminator
        # ilr, ihr = next(image_generator(config.batch_size, train_dir))
        # fake_hr = generator.predict(ilr)    
        # valid = np.ones((config.batch_size, 256, 256, 3))
        # fake = np.zero((config.batch_size, 256, 256, 3))
        # d_loss_real = discriminator.train_on_batch(ihr, valid)
        # d_loss_fake = discriminator.train_on_batch(fake_hr, fake)
        # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        ##Train generator
        # ilr, ihr = next(image_generator(config.batch_size, train_dir))
        # valid = np.ones((config.batch_size, 256, 256, 3))
        # image_features = generator.predict(ilr)
        # wandb_logging_callback = LambdaCallback(on_epoch_end=log_generator)
        
        # g_loss = gan.train_on_batch([ilr, ihr], [valid, image_features])

# print(model.summary())

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

##DONT ALTER metrics=[perceptual_distance]
# model.compile(optimizer='adam', loss='mse',
              # metrics=[perceptual_distance])

# model.fit_generator(image_generator(config.batch_size, train_dir),
                    # steps_per_epoch=config.steps_per_epoch,
                    # epochs=config.num_epochs, callbacks=[
                        # es, ImageLogger(), WandbCallback()],
                    # validation_steps=config.val_steps_per_epoch,
                    # validation_data=image_generator(config.batch_size, val_dir))
