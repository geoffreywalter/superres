import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Input, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
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
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        yield (small_images, large_images)
        counter += batch_size

#base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=[32, 32, 3], pooling=None)
# model = Sequential()
# model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same',
                         # input_shape=(config.input_width, config.input_height, 3)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(layers.Conv2D(3*8², (3, 3), activation='relu', padding='same'))
input1 = Input(shape=(32, 32, 3), dtype='float32')

conv1 = Conv2D(64, (5, 5), activation='relu', padding='same') (input1)
conv2 = Conv2D(32, (3, 3), activation='relu', padding='same') (conv1)
conv3 = Conv2D(3*8*8, (3, 3), activation='relu', padding='same') (conv2) 
lam = Lambda(lambda x: PS(x, 8))(conv3)
final = Activation('tanh')(lam)

model = Model(inputs=input1, outputs=final)
print(model.summary())


# model.add(layers.UpSampling2D())
# model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
# model.add(layers.UpSampling2D())
# model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))

# DONT ALTER metrics=[perceptual_distance]
model.compile(optimizer='adam', loss=perceptual_distance,
              metrics=[perceptual_distance])

model.fit_generator(image_generator(config.batch_size, train_dir),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                        ImageLogger(), WandbCallback()],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=image_generator(config.batch_size, val_dir))
