import random
import glob
import subprocess
import os
import gc
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Input, Activation, Lambda, BatchNormalization, Add
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, LambdaCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback
from helpfunc import PS, perceptual_distance, perceptual_distance_np
from models import SRResNet, create_generator, create_discriminator, create_gan


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
config.adversarial_epochs = 1000

config.discriminator_epochs = 1
config.discriminator_examples = 10000
config.generator_epochs = 10
config.generator_examples = 10000

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

def image_generator_gan(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        valid = np.ones((batch_size,))
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

        # wandb.log({
            # "image_generator": [wandb.Image(np.concatenate([o * 255], axis=1)) for i, o in enumerate(large_images)]
        # }, commit=False)
            
        yield (small_images, large_images, valid)
        counter += batch_size

def log_generator(epoch, logs):
    wandb.log({'generator_loss': logs['loss'],
                     'generator_perceptual_distance': logs['acc'],
                     'discriminator_loss': 0.0,
                     'discriminator_perceptual_distance': (1-logs['acc'])/2.0+0.5})

def log_discriminator(epoch, logs):
    print('discriminator_acc='+ str(logs['acc']))
    wandb.log({
            'generator_loss': 0.0,
            'generator_acc': (1.0-logs['acc'])*2.0,
            'discriminator_loss': logs['loss'],
            'discriminator_acc': logs['acc']})

            
def train_discriminator(generator, discriminator):
    imgs_lr, imgs_hr, valid = next(image_generator_gan(int(config.batch_size/2), train_dir))
    fake_hr = generator.predict(imgs_lr, batch_size=int(config.batch_size/2))

    #valid = np.ones((int(config.batch_size/2),))
    fake = np.zeros((int(config.batch_size/2),))

    imgs = np.concatenate([imgs_hr, fake_hr])
    labels = np.concatenate([valid, fake])

    #Add noise
    for label in labels:
        noise = np.random.uniform(0.0,0.3)
        if label == 0.0:
            label+= noise
        if label == 1.0:
            label-=noise
 
    wandb.log({
        "discriminator_data": [wandb.Image(np.concatenate([imgs_hr[i]*255, o * 255], axis=1)) for i, o in enumerate(fake_hr)]
    ,    "label": [o for i, o in enumerate(fake_hr)]
    }, commit=False)

    # Train the discriminators (original images = real / generated = Fake)
    d_loss_real = discriminator.train_on_batch(imgs_hr, valid)
    d_loss_fake = discriminator.train_on_batch(fake_hr, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)    
    
    print("discriminator_loss=" + str(d_loss))
    #wandb_logging_callback = LambdaCallback(on_epoch_end=log_discriminator)
    
    # discriminator.fit(imgs, labels,
        # epochs=config.discriminator_epochs,
        # batch_size=config.batch_size)
        #callbacks = [wandb_logging_callback])

    # discriminator.save(path.join(wandb.run.dir, "discriminator.h5"))

def train_generator(generator, discriminator, joint_model):
    imgs_lr, _, valid = next(image_generator_gan(config.batch_size, train_dir))

    #valid = np.ones((config.batch_size,))
    #g_loss = joint_model.train_on_batch(imgs_lr, valid)
    
    wandb_logging_callback = LambdaCallback(on_epoch_end=log_generator)

    history = joint_model.fit(imgs_lr, valid, epochs=1,
            batch_size=config.batch_size)
            #callbacks=[wandb_logging_callback])

    # generator.save(path.join(wandb.run.dir, "generator.h5"))
    
def sample_images(generator, i):
    in_sample_images, out_sample_images = next(image_generator(config.batch_size, val_dir))
    preds = generator.predict(in_sample_images)
    in_resized = []
    # Simple upsampling
    for arr in in_sample_images:
        in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
    test_img = np.zeros(
        (config.output_width, config.output_height, 3))
    wandb.log({
        "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
    }, commit=False)
    
    ##perceptual_distance
    # perc = 0
    # if (i%100 == 1):
        # for n in range(config.steps_per_epoch):
            # in_sample_images, out_sample_images = next(image_generator(config.batch_size, train_dir))
            # preds = generator.predict(in_sample_images)
            # perc += perceptual_distance_np(out_sample_images, preds)
            
        # print("Perceptual_distance=" + str(perc/config.steps_per_epoch))
        # wandb.log({"perceptual_distance": str(perc/config.steps_per_epoch)}, commit=False)
        
    
def train():
    #init
    discriminator = create_discriminator()
    generator     = create_generator()    
    #generator.load_weights('gen_model.h5')
    joint_model   = create_gan(generator, discriminator)
    generator.summary()
    discriminator.summary()
    
    for i in range(config.adversarial_epochs):
        train_discriminator(generator, discriminator)
        for j in range(config.generator_epochs):
            train_generator(generator, discriminator, joint_model)

        if (i%10 == 1):
            print("============================")
            print("============================")
            print("======i="+str(i))
            print("============================")
            print("============================")
            sample_images(generator, i)


# # Neural network
# input1 = Input(shape=(config.input_height, config.input_width, 3), dtype='float32')
# model = Model(inputs=input1, outputs=SRResNet(input1))



# # print(model.summary())

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

# ##DONT ALTER metrics=[perceptual_distance]
# model.compile(optimizer='adam', loss='mse',
              # metrics=[perceptual_distance])

# model.fit_generator(image_generator(config.batch_size, train_dir),
                    # #steps_per_epoch=config.steps_per_epoch,
                    # steps_per_epoch=1,
                    # epochs=1, callbacks=[
                        # es, ImageLogger(), WandbCallback()],
                    # validation_steps=config.val_steps_per_epoch,
                    # validation_data=image_generator(config.batch_size, val_dir))

# model.save_weights('gen_model.h5')                    

# del model
# gc.collect()
# K.clear_session()

train()
