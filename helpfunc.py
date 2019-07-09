from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import backend as K
import glob
from PIL import Image
import numpy as np
import random
import wandb
from skimage.transform import resize
from skimage import feature

from canny_edge_detector import rgb2gray
import canny_edge_detector as ced

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
        #Data augmentation
        data_gen_args_cust = dict(#featurewise_center=True,
                        #featurewise_std_normalization=True,
                        #zca_whitening=True,
                        rotation_range=90,
                        brightness_range=(0.5, 1.0), # 0 is black, 1 is same image
                        channel_shift_range=30, # value in [-channel_shift_range, channel_shift_range] added to each channel
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        vertical_flip=True,
                        horizontal_flip=True,
                        shear_range=0.2,
                        zoom_range=0.2)
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

    while True:
        if img_dir == config.train_dir and augment:
            # sel_custom = random.randint(1, 4) == 3
            # if sel_custom:
                # image_datagen = ImageDataGenerator(**data_gen_args_cust)
                # print(data_gen_args_cust)
            # else:
            image_datagen = ImageDataGenerator(**data_gen_args)
                # print(data_gen_args)

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

            #small_image_resized = resize(large_images_augmented, (config.batch_size, config.input_width, config.input_height, 3), preserve_range=True, order=1, anti_aliasing=False)
            # if sel_custom:
                # small_images_augmented = resize(large_images_augmented, (config.batch_size, config.input_width, config.input_height, 3), preserve_range=True, order=1, anti_aliasing=False)

            # totalerr = 0
            # for i in range(config.batch_size):
                # err = np.sum((small_images[i].astype("float") - small_image_resized[i].astype("float")) ** 2)
                # err /= float(small_images[i].shape[0] * small_images[i].shape[1])
                # totalerr += err
            # print("===Resizing difference=" + str(totalerr/config.batch_size))

            if counter == 0:
                augment = [np.concatenate([large_images[i], denormalize(large_images_augmented[i], config.norm0)], axis=1) for i in range(5)]
                augment_con = np.transpose(np.concatenate(augment), axes=(0, 1, 2))
                wandb.log({
                    "augment": [wandb.Image(augment_con)]
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
        in_sample_images, out_sample_images = next(image_generator(7, config.val_dir, config, shuffle=True))
        preds = self.model.predict(in_sample_images)
        # Simple upsampling
        in_resized = [in_sample_images[i].repeat(8, axis=0).repeat(8, axis=1) for i in range(len(in_sample_images))]

        # To see predictions on train set
        in_sample_images_train, out_sample_images_train = next(image_generator(7, config.train_dir, config, shuffle=True, augment=False))
        preds_train = self.model.predict(in_sample_images_train)
        in_resized_train = [in_sample_images_train[i].repeat(8, axis=0).repeat(8, axis=1) for i in range(len(in_sample_images_train))]

        # To see learning evolution on a test image
        img_lr = np.zeros((5, config.input_width, config.input_height, 3))
        img_hr = np.zeros((5, config.output_width, config.output_height, 3))
        img_name = ["data/test/4738140013-rose-in.jpg", "data/test/35869417191-hydrangea-in.jpg", "data/test/35825252922-orchid-in.jpg", "data/test/7503047224-daisy-in.jpg", "data/test/flowers-petals-plants-39517-in.jpg"]
        for i in range(len(img_name)):
            img_lr[i] = normalize(np.array(Image.open(img_name[i])), config.norm0)
            img_hr[i] = normalize(np.array(Image.open(img_name[i].replace("-in.jpg", "-out.jpg"))), config.norm0)
        preds_learn = self.model.predict(img_lr)
        in_resized_learn = [img_lr[i].repeat(8, axis=0).repeat(8, axis=1) for i in range(len(img_name))]

        # Output log formatting
        out_pred      = [np.concatenate([denormalize(in_resized[i], config.norm0), denormalize(o, config.norm0), denormalize(out_sample_images[i], config.norm0)], axis=1) for i, o in enumerate(preds)]
        img_pred_con  = np.transpose(np.concatenate(out_pred), axes=(0, 1, 2))
        out_train     = [np.concatenate([denormalize(in_resized_train[i], config.norm0), denormalize(o, config.norm0), denormalize(out_sample_images_train[i], config.norm0)], axis=1) for i, o in enumerate(preds_train)]
        img_train_con = np.transpose(np.concatenate(out_train), axes=(0, 1, 2))
        out_learn     = [np.concatenate([denormalize(in_resized_learn[i], config.norm0), denormalize(o, config.norm0), denormalize(img_hr[i], config.norm0)], axis=1) for i, o in enumerate(preds_learn)]
        img_learn_con = np.transpose(np.concatenate(out_learn), axes=(0, 1, 2))

        # #Create gif for training
        # if epoch == 0:
            # Image.fromarray(img_learn_con.astype("uint8")).save('train.gif', format='GIF', save_all=True, duration=500, loop=0)
        # else:
            # gif_pil = Image.open('train.gif')
            # gif_frames = []
            # for frame in range(0, gif_pil.n_frames):
                # gif_pil.seek(frame)
                # gif_frames.append([np.array(gif_pil), gif_pil.getpalette()])
            # print("len gif_frames=" + str(len(gif_frames)))

            # # gif_frames_pil = [Image.fromarray(gif_frames[i].astype("uint8")) for i in range(len(gif_frames))]
            # gif_frames_pil = []
            # for f, palette in gif_frames:
                # image = Image.fromarray(f.astype("uint8"))
                # image.putpalette(palette)
                # gif_frames_pil.append(image)

            # gif_frames_pil.append(Image.fromarray(img_learn_con.astype("uint8")))
            # gif_frames_pil[0].save('train.gif', format='GIF', append_images=gif_frames_pil[1:], save_all=True, duration=500, loop=0)


        wandb.log({
            "predict": [wandb.Image(img_pred_con)]
        ,   "train":   [wandb.Image(img_train_con)]
        ,   "learn":   [wandb.Image(img_learn_con)]
        }, commit=False)

class ImageLoggerCanny(Callback):
    def __init__(self, config):
        self.config = config
    def on_epoch_end(self, epoch, logs):
        config = self.config
        hr_img, edge_soft = next(image_generator_canny(5, config.val_dir, config))
        preds = self.model.predict(hr_img)

        # Output log formatting
        out = np.concatenate([hr_img[i] for i in range(5)])
        canny = np.concatenate([preds[i] for i in range(5)])
        canny_soft = np.concatenate([edge_soft[i] for i in range(5)])

        wandb.log({
             "hr": [wandb.Image(out)]
        ,    "canny": [wandb.Image(canny)]
        ,    "canny_soft": [wandb.Image(canny_soft)]
        }, commit=False)

def image_generator_canny(batch_size, img_dir, config):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-out.jpg")
    counter = 0
    random.shuffle(input_filenames)
    while True:

        hr_img = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        edge_soft = np.zeros(
            (batch_size, config.output_width, config.output_height, 1))
        edge_soft_gray = np.zeros(
            (batch_size, config.output_width, config.output_height, 1))
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            hr_img[i] = np.array(Image.open(img))
            edge_soft_gray[i] = rgb2gray(hr_img[i])

        detector = ced.cannyEdgeDetector(edge_soft_gray, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
        edge_soft_list = detector.detect()

        for i in range(batch_size):
            edge_soft[i] = edge_soft_list[i]

        if counter == 0 and img_dir == config.train_dir:
            out = np.concatenate([hr_img[i] for i in range(5)])
            canny = np.concatenate([edge_soft[i] for i in range(5)])

            wandb.log({
                 "hr": [wandb.Image(out)]
            ,    "canny": [wandb.Image(canny)]
            }, commit=False)

        hr_img    = normalize(hr_img   , config.norm0)
        edge_soft = normalize(edge_soft, config.norm0)


        yield (hr_img, edge_soft)
        counter += batch_size
