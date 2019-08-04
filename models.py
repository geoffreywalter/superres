from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Input, Activation, Lambda, BatchNormalization, Add, Dense, Flatten, LeakyReLU, Concatenate, GaussianNoise, Lambda, MaxPooling2D, AveragePooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from helpfunc import PS, perceptual_distance, custom_loss
from canny_edge_detector import rgb2gray
import canny_edge_detector as ced
import tensorflow as tf

def resBlock(tens, filter_size):
    x = Conv2D(filter_size, (3, 3), padding='same') (tens)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter_size, (3, 3), padding='same') (x)
    x = BatchNormalization()(x)
    x = Add()([x, tens])
    x = Activation('relu')(x)
    return x

def SRResNet(input, filters, nBlocks):
    skipRes = x = Conv2D(filters, (9, 9), activation='relu', padding='same') (input)

    # Residual blocks
    for i in range(nBlocks):
        x = resBlock(x, filters)

    x = Conv2D(filters, (3, 3), padding='same') (x)
    x = BatchNormalization()(x)
    x = Add()([skipRes, x])

    # Sub-pixel convolution layer
    r = 8 #Upscale x8
    x = Conv2D(3*r*r, (3, 3), padding='same') (x)
    x = Lambda(lambda x: PS(x, r))(x)

    x = Conv2D(3, (9, 9), activation='tanh', padding='same') (x)
    return x

def EDSRBlock(tens, filter_size):
    x = Conv2D(filter_size, (3, 3), padding='same') (tens)
    # x = LeakyReLU(alpha=0.1)(x)
    x = Activation('relu')(x)
    x = Conv2D(filter_size, (3, 3), padding='same') (x)
    x = Lambda(lambda x: x * 0.1)(x)
    x = Add()([x, tens])
    return x

def EDSR(input, filters, nBlocks):
    skipRes = x = Conv2D(filters, (3, 3), padding='same') (input)

    # Residual blocks
    for i in range(nBlocks):
        x = EDSRBlock(x, filters)

    x = Conv2D(filters, (3, 3), padding='same') (x)
    x = Add()([skipRes, x])

    # Sub-pixel convolution layer
    r = 8 #Upscale x8
    x = Conv2D(3*r*r, (3, 3), padding='same') (x)
    x = Lambda(lambda x: PS(x, r))(x)

    x = Conv2D(3, (3, 3), activation='tanh', padding='same') (x)
    return x

def canny(input):
    edge_gray = rgb2gray(input)
    detector = ced.cannyEdgeDetector(edge_gray, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
    return detector.detect()

def Canny(input):
    x = Lambda(lambda x: canny(x)) (input)
    return x

def Merge(inputs, filters, nBlocks):
    edsr = inputs[0]
    edge = inputs[1]

    x = Concatenate()([edsr, edge])
    edsr = x = Conv2D(filters, (3, 3), padding='same') (x)

    # Residual blocks
    for i in range(nBlocks):
        x = EDSRBlock(x, filters)

    x = Conv2D(filters, (3, 3), padding='same') (x)
    edge = Conv2D(filters, (3, 3), padding='same') (edge)
    x = Add()([edge, edsr, x])

    x = Conv2D(3, (3, 3), activation='tanh', padding='same') (x)
    return x

def Attention(input, filters, nBlocks, nLayers):
    bicubic = Lambda(lambda x: tf.image.resize_bicubic(x, (256, 256), align_corners=True)) (input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same') (bicubic)
    skip = x = Conv2D(64, (3, 3), activation='relu', padding='same') (x)

    x = MaxPooling2D((2, 2), 2) (x)
    skipDense1 = x = DenseBlock(x, filters, nLayers)
    x = AveragePooling2D((2, 2), 2) (x)
    skipDense2 = x = DenseBlock(x, filters, nLayers)
    x = AveragePooling2D((2, 2), 2) (x)
    x = DenseBlock(x, filters, nLayers)

    # x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same') (x)
    # x = Concatenate()([x, skipDense2])
    # x = DenseBlock(x, filters, nLayers)
    # x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same') (x)
    # x = Concatenate()([x, skipDense1])
    # x = DenseBlock(x, filters, nLayers)
    # x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same') (x)

    r = 2
    x = Conv2D(3*r*r, (3, 3), padding='same') (x)
    x = Lambda(lambda x: PS(x, r))(x)

    x = Concatenate()([x, skipDense2])
    x = DenseBlock(x, filters, nLayers)
    x = Conv2D(3*r*r, (3, 3), padding='same') (x)
    x = Lambda(lambda x: PS(x, r))(x)
    x = Concatenate()([x, skipDense1])
    x = DenseBlock(x, filters, nLayers)
    x = Conv2D(3*r*r, (3, 3), padding='same') (x)
    x = Lambda(lambda x: PS(x, r))(x)

    x = Concatenate()([x, skip])
    x = Conv2D(64, (3, 3), activation='relu', padding='same') (x)
    x = Conv2D(1, (1, 1), activation='sigmoid', padding='same') (x)

    return x

def DenseBlock(tens, filter_size, nLayers):
    x = Conv2D(filter_size, (3, 3), padding='same') (tens)
    x = Activation('relu')(x)

    for i in range(nLayers):
        mid = Conv2D(filter_size, (3, 3), activation='relu', padding='same') (x)
        x = Concatenate()([x, mid])
    return x

def SRDenseNet(input, filters, nBlocks, nLayers):
    x = Conv2D(128, (3, 3), padding='same') (input)
    con = x = Activation('relu')(x)

    # Dense blocks
    for i in range(nBlocks):
        x = DenseBlock(x, filters, nLayers)
        con = Concatenate()([x, con])

    # BottleNeck Layer
    x = Conv2D(512, (1, 1), padding='same') (con)
    x = Activation('relu')(x)

    # Sub-pixel convolution layer
    r = 8 #Upscale x8
    x = Conv2D(3*r*r, (3, 3), padding='same') (x)
    x = Lambda(lambda x: PS(x, r))(x)

    x = Conv2D(3, (3, 3), activation='tanh', padding='same') (x)
    return x

def create_generator():
    input = Input(shape=(32, 32, 3))
    model = Model(inputs=input, outputs=EDSR(input, 128, 32))
   #model.compile(loss='mse', optimizer='adam', metrics=[perceptual_distance])
    return model

def create_discriminator():
    input = Input(shape=(256, 256, 3))

    x = Conv2D(64, (3, 3), padding='same') (input)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(64, (3, 3), strides=2, padding='same') (x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (3, 3), strides=1, padding='same') (x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (3, 3), strides=2, padding='same') (x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (3, 3), strides=1, padding='same') (x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (3, 3), strides=2, padding='same') (x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, (3, 3), strides=1, padding='same') (x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, (3, 3), strides=2, padding='same') (x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(1024) (x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(1) (x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=input, outputs=x, name="discriminator")
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-4, 0.9))
    return model

def create_gan(generator, discriminator):
    discriminator.trainable=False
    gan_input = Input(shape=(32, 32, 3))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output], name="gan")
    loss_funcs = {"generator": custom_loss, "discriminator":'binary_crossentropy'}
    loss_weights = {"generator":1., "discriminator":1e-3}
    metrics = {"generator":perceptual_distance}
    gan.compile(loss=loss_funcs, loss_weights=loss_weights, optimizer=Adam(1e-5, 0.9), metrics=metrics)
    return gan
