import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import img_to_array
import wandb
from wandb.keras import WandbCallback

os.environ["WANDB_MODE"] = "dryrun"
run = wandb.init(project='superres')
config = run.config

config.num_epochs = 1000
config.batch_size = 32
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256

val_dir = 'test'
train_dir = 'train'

# automatically get the data if it doesn't exist
if not os.path.exists("train"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz", shell=True)

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size


def image_generator(batch_size, img_dir):
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
            small_images[i] = img_to_array(Image.open(img))
            large_images[i] = img_to_array(
                Image.open(img.replace("-in.jpg", "-out.jpg")))
        yield (small_images, large_images)
        counter += batch_size


def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


model = Sequential()
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same',
                        input_shape=(config.input_width, config.input_height, 3)))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))

# DONT ALTER metrics=[perceptual_distance]
model.compile(optimizer='adam', loss='mse',
              metrics=[perceptual_distance])

val_generator = image_generator(config.batch_size, val_dir)
model.fit_generator(image_generator(config.batch_size, train_dir),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                        WandbCallback(data_type='image', generator=val_generator, predictions=16)],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=val_generator)
