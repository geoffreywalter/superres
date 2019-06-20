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
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback
from helpfunc import PS, perceptual_distance


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

def image_generator(batch_size, img_dir, augment=True):
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

#base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=[32, 32, 3], pooling=None)
# model = Sequential()
# model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same',
                         # input_shape=(config.input_width, config.input_height, 3)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(layers.Conv2D(3*8², (3, 3), activation='relu', padding='same'))

def resBlock(tens, filter_size):
    x = Conv2D(filter_size, (3, 3), padding='same') (tens)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter_size, (3, 3), padding='same') (x)
    x = BatchNormalization()(x)
    x = Add()([x, tens])
    x = Activation('relu')(x)
    return x

input1 = Input(shape=(32, 32, 3), dtype='float32')

x = Conv2D(64, (5, 5), activation='relu', padding='same') (input1)
temp = x

# Resnet layers
for i in range(16):
    x = resBlock(x, 64)

x = Conv2D(64, (3, 3), activation='relu', padding='same') (x) 
x = BatchNormalization()(x)
x = Add()([temp, x])

# Sub-pixel convolution layer 1
r = 4
x = Conv2D(3*r*r, (3, 3), activation='relu', padding='same') (x) 
x = Lambda(lambda x: PS(x, r))(x)
x = Activation('tanh')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same') (x) 
# Sub-pixel convolution layer 2
r = 2
x = Conv2D(3*r*r, (3, 3), activation='relu', padding='same') (x) 
x = Lambda(lambda x: PS(x, r))(x)
x = Activation('tanh')(x)

x = Conv2D(3, (9, 9), activation='relu', padding='same') (x) 

model = Model(inputs=input1, outputs=x)
print(model.summary())

# DONT ALTER metrics=[perceptual_distance]
model.compile(optimizer='adam', loss='mse',
              metrics=[perceptual_distance])

model.fit_generator(image_generator(config.batch_size, train_dir),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                        ImageLogger(), WandbCallback()],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=image_generator(config.batch_size, val_dir))
