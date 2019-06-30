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
from helpfunc import PS, perceptual_distance, perceptual_distance_np, image_generator
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
config.adversarial_epochs = 20000

config.discriminator_epochs = 2
config.discriminator_examples = 10000
config.generator_epochs = 1
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
        in_sample_images, out_sample_images = next(image_generator(5, val_dir, config, shuffle=False))
        preds = self.model.predict(in_sample_images)
        in_resized = []
        # Simple upsampling
        for arr in in_sample_images:
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        test_img = np.zeros(
            (config.output_width, config.output_height, 3))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([(in_resized[i] + 1.0) * 127.5, (o + 1.0) * 127.5, (out_sample_images[i] + 1.0) * 127.5], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)

def image_generator_gan(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    random.shuffle(input_filenames)
    while True:
        valid = np.ones((batch_size,))
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 127.5 - 1.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 127.5 - 1.0

        # wandb.log({
            # "image_generator": [wandb.Image(np.concatenate([o * 255], axis=1)) for i, o in enumerate(large_images)]
        # }, commit=True)
            
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

            
def train_discriminator(generator, discriminator, i):
    imgs_lr, imgs_hr, valid = next(image_generator_gan(int(config.batch_size), train_dir))
    if i%2:
        fake_hr = generator.predict(imgs_lr, batch_size=int(config.batch_size))

    #valid = np.ones((int(config.batch_size/2),))
        fake = np.zeros((int(config.batch_size),))
        imgs = np.concatenate([fake_hr])
        labels = np.concatenate([fake])

    else:
    #valid = np.ones((int(config.batch_size/2),))
        imgs = np.concatenate([imgs_hr])
        labels = np.concatenate([valid])


    #Add noise
    for label in labels:
        noise = np.random.uniform(0.0,0.3)
        if label == 0.0:
            label+= noise
        if label == 1.0:
            label-=noise
 
    # wandb.log({
        # "discriminator_data": [wandb.Image(np.concatenate([imgs_hr[i]*255, o * 255], axis=1)) for i, o in enumerate(fake_hr)]
    # })

    # Train the discriminators (original images = real / generated = Fake)
    d_loss = discriminator.train_on_batch(imgs, labels)

  #  d_loss_fake = discriminator.train_on_batch(fake_hr, fake)
    #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)    
    
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
    in_sample_images, out_sample_images = next(image_generator_sample(5, val_dir, config, shuffle=True))
    preds = generator.predict(in_sample_images)
    in_resized = []
    # Simple upsampling
    for arr in in_sample_images:
        in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
    test_img = np.zeros(
        (config.output_width, config.output_height, 3))
    wandb.log({
        "examples": [wandb.Image(np.concatenate([(in_resized[i]+1.0) * 127.5, (o+1.0) * 127.5, (out_sample_images[i]+1.0) * 127.5], axis=1)) for i, o in enumerate(preds)]
    })
    
    #perceptual_distance
    perc = 0
    if (i%400 == 1):
        #for n in range(config.steps_per_epoch):
        in_sample_images, out_sample_images = next(image_generator(config.batch_size, train_dir, config, shuffle=False))
        preds = generator.predict(in_sample_images)
        perc += perceptual_distance_np(out_sample_images, preds)
            
        print("Perceptual_distance=" + str(perc))
        wandb.log({"perceptual_distance": str(perc)}, commit=False)
        # print("Perceptual_distance=" + str(perc/config.steps_per_epoch))
        # wandb.log({"perceptual_distance": str(perc/config.steps_per_epoch)}, commit=False)
        
    
def train():
    #init
    discriminator = create_discriminator()
    generator     = create_generator()    
    generator.load_weights('srresnet.h5')
    joint_model   = create_gan(generator, discriminator)
    generator.summary()
    discriminator.summary()
    
    #Init discriminator training
    for j in range(1000):
        train_discriminator(generator, discriminator, j)
    
    for i in range(config.adversarial_epochs):
        for j in range(config.discriminator_epochs):
            train_discriminator(generator, discriminator, i)
        for j in range(config.generator_epochs):
            train_generator(generator, discriminator, joint_model)

        if (i%40 == 1):
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
