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
from tensorflow.keras.layers import Conv2D, Input, Activation, Lambda, BatchNormalization, Add, Dot, Multiply
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, LambdaCallback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import wandb
from wandb.keras import WandbCallback
from helpfunc import PS, perceptual_distance, perceptual_distance_np, image_generator, ImageLogger
from models import EDSR, SRDenseNet, Attention


#GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 80
config.batch_size = 16
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256
config.norm0 = True
config.name = "AttNet"
config.reconstructionNN = "EDSR"
config.filters = 128
config.nBlocks = 32
config.custom_aug = False
config.Att_filters = 16
config.Att_nBlocks = 10
config.Att_nLayers = 8

# config.filters = 16
# config.nBlocks = 8
# config.nLayers = 8

config.val_dir = 'data/test'
config.train_dir = 'data/train'

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)

config.steps_per_epoch = len(
    glob.glob(config.train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(config.val_dir + "/*-in.jpg")) // config.batch_size

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
    attention_out = attention(input)

    out1 = Multiply() ([reconstruction_out, attention_out])
    out2 = Add() ([out1, bicubic])
    #out3 = Lambda(lambda x: K.clip(x, 0.0, 1.0)) (out2)

    model = Model(inputs=input, outputs=out2)
    adam = Adam(epsilon=0.1)
    model.compile(optimizer=adam, loss=[perceptual_distance], metrics=[perceptual_distance])
    return model

# Neural network
reconstruction = create_reconstruction()
attention = create_attention()
model = create_merge(reconstruction, attention)

# print(reconstruction.summary())
# print("^^^ reconstruction ^^^")
# print(attention.summary())
# print("^^^ attention ^^^")
print(model.summary())
print("^^^ model ^^^")
#model.load_weights('srdensenet.h5')

#es = EarlyStopping(monitor='val_perceptual_distance', mode='min', verbose = 1, patience=2)
mc = ModelCheckpoint('attnet.h5', monitor='val_perceptual_distance', mode='min', save_best_only=True)

# adam = Adam(clipvalue=7500.0)
##DONT ALTER metrics=[perceptual_distance]

model.fit_generator(image_generator(config.batch_size, config.train_dir, config),
                    steps_per_epoch=1,
                    epochs=config.num_epochs, callbacks=[
                        mc, ImageLogger(config, reconstruction, attention), WandbCallback()],
                    validation_steps=1,
                    validation_data=image_generator(config.batch_size, config.val_dir, config))