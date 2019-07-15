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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback
from helpfunc import PS, perceptual_distance, perceptual_distance_np, image_generator, ImageLogger
from models import EDSR, Canny, Merge


#GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 80
config.batch_size = 10
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256
config.norm0 = True
config.name = "SREdgeNet"
config.EDSR_filters = 128
config.EDSR_nBlocks = 32
config.Merge_filters = 64
config.Merge_nBlocks = 8

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

def create_edsr():
    input = Input(shape=(32, 32, 3))
    model = Model(inputs=input, outputs=EDSR(input, config.EDSR_filters, config.EDSR_nBlocks))
    model.load_weights('sredgenet_edsr.h5')
    #model.compile(optimizer='adam', loss=[perceptual_distance], metrics=[perceptual_distance])
    return model

def create_edge():
    input = Input(shape=(256, 256, 3))
    model = Model(inputs=input, outputs=Canny(input)) #Fixed filters and nBlocks because of load weights
    return model

def create_merge():
    input_edsr = Input(shape=(256, 256, 3))
    input_canny = Input(shape=(256, 256, 1))
    model = Model(inputs=[input_edsr, input_canny], outputs=Merge([input_edsr, input_canny], config.Merge_filters, config.Merge_nBlocks))
    return model

def create_sredgenet(edsr, canny, merge):
    canny.trainable=False
    # edsr.trainable=False
    input = Input(shape=(32, 32, 3))
    edsr = edsr(input)
    canny = canny(edsr)
    merge = merge([edsr, canny])
    model = Model(inputs=input, outputs=merge)
    model.compile(optimizer='adam', loss=[perceptual_distance], metrics=[perceptual_distance])
    return model

def create_edsr_edge(edsr, canny):
    input = Input(shape=(32, 32, 3))
    edsr = edsr(input)
    canny = canny(edsr)
    model = Model(inputs=input, outputs=canny)
    return model

# Neural network
edsr = create_edsr()
edge = create_edge()
merge = create_merge()
sredgenet = create_sredgenet(edsr, edge, merge)

# Model to log edges
edsr_edge = create_edsr_edge(edsr, edge)

print(sredgenet.summary())
#model.load_weights('edsr.h5')

mc = ModelCheckpoint('sredgenet.h5', monitor='val_perceptual_distance', mode='min', save_best_only=True)

sredgenet.fit_generator(image_generator(config.batch_size, config.train_dir, config),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                        mc, ImageLogger(config, edsr_edge), WandbCallback()],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=image_generator(config.batch_size, config.val_dir, config))
