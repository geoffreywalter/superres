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
from tensorflow.keras.layers import Conv2D, Input, Activation, Lambda, BatchNormalization, Add, Dot, Multiply, Concatenate, Reshape, Cropping2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, LambdaCallback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import wandb
from wandb.keras import WandbCallback
from helpfunc import PS, perceptual_distance, perceptual_distance_np, image_generator, ImageLogger, normalize, denormalize
from models import EDSR, SRDenseNet, Attention, create_generator, create_discriminator, create_gan

#GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

run = wandb.init(project='superres')
config = run.config

config.batch_size = 16
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256
config.norm0 = True
config.name = "GAN"
config.reconstructionNN = "EDSR"
config.filters = 128
config.nBlocks = 16
config.custom_aug = False
config.Att_filters = 16
config.Att_nBlocks = 10
config.Att_nLayers = 8

config.val_dir = 'data/test'
config.train_dir = 'data/train'

config.adversarial_epochs = 20000
config.discriminator_epochs = 2
config.generator_epochs = 1

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)

def create_reconstruction():
    input = Input(shape=(32, 32, 3))
    model = Model(inputs=input, outputs=EDSR(input, config.filters, config.nBlocks))
    return model

def create_attention():
    input = Input(shape=(32, 32, 3))
    model = Model(inputs=input, outputs=Attention(input, config.Att_filters, config.Att_nBlocks, config.Att_nLayers))
    return model

def create_merge(reconstruction, attention):
    input = Input(shape=(32, 32, 3))
    bicubic = Lambda(lambda x: tf.image.resize_bicubic(x, (256, 256), align_corners=True)) (input)

    reconstruction_out = reconstruction(input)
    #attention_out = attention(input)

    # out1 = Lambda(lambda x: x[0] * x[1])([reconstruction_out, attention_out])
    # out2 = Add() ([out1, bicubic])

    out2 = Add() ([reconstruction_out, bicubic])
    model = Model(inputs=input, outputs=out2, name="generator")
    # adam = Adam(epsilon=0.1)#, lr=0.001)#, decay=(1/2)**(1.0/100.0))
    #model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])
    return model

# Neural network
reconstruction = create_reconstruction()
attention = create_attention()
merge = create_merge(reconstruction, attention)

#model.load_weights('attnet_266.h5')

mc = ModelCheckpoint('gan.h5', monitor='val_perceptual_distance', mode='min', save_best_only=True)

config.steps_per_epoch = len(
    glob.glob(config.train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(config.val_dir + "/*-in.jpg")) // config.batch_size

def log_generator(epoch, logs):
    wandb.log({'generator_perceptual_distance': logs['perceptual_distance']})

def log_discriminator(epoch, logs):
    #print('discriminator_acc='+ str(logs['acc']))
    wandb.log({
            'generator_loss': 0.0,
            'generator_acc': (1.0-logs['acc'])*2.0,
            'discriminator_loss': logs['loss'],
            'discriminator_acc': logs['acc']})


def train_discriminator(generator, discriminator, i):
    imgs_lr, imgs_hr = next(image_generator(config.batch_size, config.train_dir, config))
    if i%2:
        fake_hr = generator.predict(imgs_lr, batch_size=config.batch_size)

    #valid = np.ones((int(config.batch_size/2),))
        fake = np.zeros((config.batch_size,))
        imgs = np.concatenate([fake_hr])
        labels = np.concatenate([fake])

    else:
        valid = np.ones((config.batch_size,))
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

    #print("discriminator_loss=" + str(d_loss))
    #wandb_logging_callback = LambdaCallback(on_epoch_end=log_discriminator)

    # discriminator.fit(imgs, labels,
        # epochs=config.discriminator_epochs,
        # batch_size=config.batch_size)
        #callbacks = [wandb_logging_callback])

    # discriminator.save(path.join(wandb.run.dir, "discriminator.h5"))

def train_generator(generator, discriminator, joint_model):
    imgs_lr, imgs_hr = next(image_generator(config.batch_size, config.train_dir, config))

    valid = np.ones((config.batch_size,))
    #g_loss = joint_model.train_on_batch(imgs_lr, valid)

    wandb_logging_callback = LambdaCallback(on_epoch_end=log_generator)
    #mc = ModelCheckpoint('generator_gan.h5', monitor='val_perceptual_distance', mode='min', save_best_only=True)
    y_train = {"generator": imgs_hr, "discriminator": valid}
    history = joint_model.fit(imgs_lr, y_train, epochs=1,
            batch_size=config.batch_size)
            #callbacks=[mc])

    # generator.save(path.join(wandb.run.dir, "generator.h5"))

def sample_images(generator, i):
    if i%100==99:
        generator.save_weights("generator.h5")
        # To see learning evolution on a test image
        img_lr = np.zeros((5, config.input_width, config.input_height, 3))
        img_hr = np.zeros((5, config.output_width, config.output_height, 3))
        img_name = ["data/test/4738140013-rose-in.jpg", "data/test/35869417191-hydrangea-in.jpg", "data/test/35825252922-orchid-in.jpg", "data/test/7503047224-daisy-in.jpg", "data/test/flowers-petals-plants-39517-in.jpg"]
        for i in range(len(img_name)):
            img_lr[i] = normalize(np.array(Image.open(img_name[i])), config.norm0)
            img_hr[i] = normalize(np.array(Image.open(img_name[i].replace("-in.jpg", "-out.jpg"))), config.norm0)
        preds_learn = generator.predict(img_lr)
        in_resized_learn = [img_lr[i].repeat(8, axis=0).repeat(8, axis=1) for i in range(len(img_name))]
        out_learn     = [np.concatenate([denormalize(in_resized_learn[i], config.norm0), denormalize(o, config.norm0), denormalize(img_hr[i], config.norm0)], axis=1) for i, o in enumerate(preds_learn)]
        img_learn_con = np.transpose(np.concatenate(out_learn), axes=(0, 1, 2))
        img_learn_con_pil = Image.fromarray(np.clip(img_learn_con, 0, 255).astype("uint8"), mode='RGB')
        wandb.log({
            "learn":   [wandb.Image(img_learn_con_pil)]
        }, commit=False)

    #perceptual_distance
    #if (i%400 == 1):
        perc = 0
        for n in range(config.steps_per_epoch):
            in_sample_images, out_sample_images = next(image_generator(config.batch_size, config.val_dir, config))
            preds = generator.predict(in_sample_images)
            perc += perceptual_distance_np(out_sample_images, preds)

            #print("Perceptual_distance=" + str(perc))
            #wandb.log({"perceptual_distance": str(perc)}, commit=False)
        print("Perceptual_distance=" + str(perc/config.steps_per_epoch))
        wandb.log({'val_perceptual_distance': perc/config.steps_per_epoch
        , 'epoch': int(i/100.0)
        }, commit=True)


def train():
    #init
    discriminator = create_discriminator()
    generator     = merge
    #generator.load_weights('srresnet.h5')
    joint_model   = create_gan(generator, discriminator)
    joint_model.load_weights('srgan.h5')
    #generator.summary()
    #discriminator.summary()

    #Init discriminator training
    # for j in range(1000):
    #     train_discriminator(generator, discriminator, j)

    for i in range(config.adversarial_epochs):
        for j in range(config.discriminator_epochs):
            discriminator.trainable = True
            train_discriminator(generator, discriminator, i)
            discriminator.trainable = False
        for j in range(config.generator_epochs):
            train_generator(generator, discriminator, joint_model)

        joint_model.save_weights('srgan.h5')
        sample_images(generator, i)

train()
