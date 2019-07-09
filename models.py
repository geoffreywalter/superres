from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Input, Activation, Lambda, BatchNormalization, Add, Dense, Flatten, LeakyReLU, Concatenate, GaussianNoise
from tensorflow.keras.optimizers import Adam
from helpfunc import PS, perceptual_distance

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

def Canny(input, filters, nBlocks):
    x = Conv2D(128, (3, 3), activation='relu', padding='same') (input)

    # Residual blocks
    for i in range(nBlocks):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same') (x)

    x = Conv2D(1, (3, 3), activation='tanh', padding='same') (x)

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

def DenseBlock(tens, filter_size, nLayers):
    x = Conv2D(filter_size, (3, 3), padding='same') (tens)
    x = Activation('relu')(x)

    for i in range(nLayers):
        mid = Conv2D(filter_size, (3, 3), activation='relu', padding='same') (x)
        x = Concatenate()([x, mid])
    return x

def SRDenseNet(input, filters, nBlocks, nLayers):
    x = Conv2D(256, (3, 3), padding='same') (input)
    con = x = Activation('relu')(x)

    # Dense blocks
    for i in range(nBlocks):
        x = DenseBlock(x, filters, nLayers)
        con = Concatenate()([x, con])

    # # BottleNeck Layer
    # x = Conv2D(512, (1, 1), padding='same') (con)
    # x = Activation('relu')(x)

    # Sub-pixel convolution layer
    r = 8 #Upscale x8
    x = Conv2D(3*r*r, (3, 3), padding='same') (con)
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

    #normalization from [-1, 1] to [0, 1]
    x = Lambda(lambda x: (x + 1.0) / 2.0) (input)

    x = Conv2D(64, (3, 3), padding='same') (x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(64, (3, 3), strides=2, padding='same') (input)
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

    model = Model(inputs=input, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-4, 0.9))
    return model

def create_gan(generator, discriminator):
    discriminator.trainable=False
    gan_input = Input(shape=(32, 32, 3))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(1e-5, 0.9), metrics=['acc'])
    return gan
