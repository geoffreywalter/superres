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
from tensorflow.keras.callbacks import Callback, EarlyStopping, LambdaCallback, ModelCheckpoint
import wandb
from wandb.keras import WandbCallback
from helpfunc import PS, perceptual_distance, perceptual_distance_np, image_generator_canny, ImageLoggerCanny
from models import Canny


#GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 20
config.batch_size = 32
config.input_height = 256
config.input_width = 256
config.output_height = 256
config.output_width = 256
config.norm0 = True
config.name = "Canny"
config.filters = 64
config.nBlocks = 6

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
model = Model(inputs=input1, outputs=Canny(input1, config.filters, config.nBlocks))

print(model.summary())
#model.load_weights('edsr.h5')

mc = ModelCheckpoint('canny.h5', monitor='acc', mode='min', save_best_only=True)

##DONT ALTER metrics=[perceptual_distance]
model.compile(optimizer='adam', loss='mse', metrics=["acc"])

model.fit_generator(image_generator_canny(config.batch_size, config.train_dir, config),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                        mc, ImageLoggerCanny(config), WandbCallback()])

#model.save_weights('edsr.h5')                    


