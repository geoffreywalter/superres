from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import backend as K
import glob
from PIL import Image
import numpy as np
import random
import wandb

def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    norm0 = True
    if norm0:
        y_true *= 255.0
        y_pred *= 255.0
    else:
        y_true += 1.0
        y_pred += 1.0
        y_true *= 127.5
        y_pred *= 127.5
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

def perceptual_distance_np(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255.0
    y_pred *= 255.0
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return np.mean(np.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

# _phase_shift and PS from https://github.com/tetrachrome/subpixel/blob/master/subpixel.py
def _phase_shift(I, r):
    # Helper function with main phase shift operation
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, [-1, a, b, r, r])
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2 )  # bsize, a*r, b*r
    return tf.reshape(X, [-1, a*r, b*r, 1])

def PS(X, r):
    # Main OP that you can arbitrarily use in you tensorflow code
    Xc = tf.split(X, 3, 3)
    X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    return X

def normalize(x, norm0):
    if norm0:
        return x / 255.0
    else:
        return x / 127.5 - 1.0

def denormalize(x, norm0):
    if norm0:
        return x * 255.0
    else:
        return (x + 1.0) * 127.5 
        
def image_generator(batch_size, img_dir, config, shuffle=True, augment=True):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    printed = False
    if shuffle:
        random.shuffle(input_filenames)
    if img_dir == config.train_dir and augment:
        #Data augmentation
        data_gen_args = dict(#featurewise_center=True,
                        #featurewise_std_normalization=True,
                        #zca_whitening=True,
                        #rotation_range=90,
                        brightness_range=(0.5, 1.0), # 0 is black, 1 is same image
                        channel_shift_range=30, # value in [-channel_shift_range, channel_shift_range] added to each channel
                        #width_shift_range=0.2,
                        #height_shift_range=0.2,
                        vertical_flip=True,
                        horizontal_flip=True)
                        #shear_range=0.2,
                        #zoom_range=0.2)
            
        print(data_gen_args)
        image_datagen = ImageDataGenerator(**data_gen_args)

    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        if counter+batch_size >= len(input_filenames) or not shuffle:
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img))
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg")))
        
        if img_dir == config.train_dir and augment:
            seed = random.randint(1, 100000)
            gen0 = image_datagen.flow(small_images, batch_size=config.batch_size, shuffle=False, seed=seed)
            gen1 = image_datagen.flow(large_images, batch_size=config.batch_size, shuffle=False, seed=seed)
            small_images_augmented, large_images_augmented = next(zip(gen0, gen1))
            small_images_augmented = normalize(small_images_augmented, config.norm0)
            large_images_augmented = normalize(large_images_augmented, config.norm0)
            if counter == 0:
                in_resized = []
                for arr in small_images_augmented:
                    in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
                wandb.log({
                    "augment": [wandb.Image(np.concatenate([denormalize(in_resized[i], config.norm0), denormalize(large_images_augmented[i], config.norm0)], axis=1)) for i in range(5)]
                }, commit=False)
            
            yield (small_images_augmented, large_images_augmented)        
        else:
            small_images = normalize(small_images, config.norm0)
            large_images = normalize(large_images, config.norm0)
            yield (small_images, large_images)
        counter += batch_size
        

class ImageLogger(Callback):
    def __init__(self, config):
        self.config = config
    def on_epoch_end(self, epoch, logs):
        config = self.config
        in_sample_images, out_sample_images = next(image_generator(5, config.val_dir, config, shuffle=True))
        preds = self.model.predict(in_sample_images)
        in_resized = []
        # Simple upsampling
        for arr in in_sample_images:
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        
        # See learning on train set
        in_sample_images_train, out_sample_images_train = next(image_generator(5, config.train_dir, config, shuffle=True, augment=False))
        preds_train = self.model.predict(in_sample_images_train)
        in_resized_train = []
        for arr in in_sample_images_train:
            in_resized_train.append(arr.repeat(8, axis=0).repeat(8, axis=1))

        wandb.log({
            "predict": [wandb.Image(np.concatenate([denormalize(in_resized[i], config.norm0), denormalize(o, config.norm0), denormalize(out_sample_images[i], config.norm0)], axis=1)) for i, o in enumerate(preds)]
        ,   "train": [wandb.Image(np.concatenate([denormalize(in_resized_train[i], config.norm0), denormalize(o, config.norm0), denormalize(out_sample_images_train[i], config.norm0)], axis=1)) for i, o in enumerate(preds_train)]
        }, commit=False)
