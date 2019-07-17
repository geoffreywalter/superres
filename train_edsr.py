import random
import glob
import subprocess
import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, LambdaCallback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback
from helpfunc import PS, perceptual_distance, image_generator, ImageLogger
from models import EDSR


#GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 80
config.batch_size = 32
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256
config.norm0 = True
config.custom_aug = False
config.name = "EDSR"
config.filters = 128
config.nBlocks = 16

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

# Neural network
input1 = Input(shape=(config.input_height, config.input_width, 3), dtype='float32')
model = Model(inputs=input1, outputs=EDSR(input1, config.filters, config.nBlocks))

#print(model.summary())
#model.load_weights('edsr.h5')

#es = EarlyStopping(monitor='val_perceptual_distance', mode='min', verbose = 1, patience=2)
mc = ModelCheckpoint('edsr.h5', monitor='val_perceptual_distance', mode='min', save_best_only=True)

##DONT ALTER metrics=[perceptual_distance]
model.compile(optimizer='adam', loss=[perceptual_distance], metrics=[perceptual_distance])

model.fit_generator(image_generator(config.batch_size, config.train_dir, config),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                        mc, ImageLogger(config), WandbCallback()],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=image_generator(config.batch_size, config.val_dir, config))

#model.save_weights('edsr.h5')
