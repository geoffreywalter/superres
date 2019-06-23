from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Input, Activation, Lambda, BatchNormalization, Add, Dense, Flatten
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

def SRResNet(input):
    x = Conv2D(64, (5, 5), activation='relu', padding='same') (input)
    skipRes = x

    # Resnet layers
    for i in range(2):
        x = resBlock(x, 64)

    x = Conv2D(64, (3, 3), activation='relu', padding='same') (x) 
    x = BatchNormalization()(x)
    x = Add()([skipRes, x])

    # Sub-pixel convolution layer 1
    r = 8 #Upscale x4
    x = Conv2D(3*r*r, (3, 3), activation='relu', padding='same') (x) 
    x = Lambda(lambda x: PS(x, r))(x)
    x = Activation('tanh')(x)

    #x = Conv2D(3, (9, 9), activation='relu', padding='same') (x) 
    return x
    
def create_generator():
    input = Input(shape=(32, 32, 3))
    model = Model(inputs=input, outputs=SRResNet(input))
    #model.compile(loss='mse', optimizer='adam', metrics=[perceptual_distance])
    return model

def create_discriminator():
    input = Input(shape=(256, 256, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same') (input)

    x = Conv2D(64, (3, 3), strides=2, padding='same') (x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #x = Conv2D(128, (3, 3), strides=1, padding='same') (x) 
    x = Conv2D(32, (3, 3), strides=2, padding='same') (x) 
    x = Conv2D(32, (3, 3), strides=2, padding='same') (x) 
    x = Conv2D(32, (3, 3), strides=2, padding='same') (x) 
    #x = Conv2D(256, (3, 3), strides=1, padding='same') (x) 
    #x = Conv2D(256, (3, 3), strides=2, padding='same') (x) 
    #x = Conv2D(512, (3, 3), strides=1, padding='same') (x) 
    #x = Conv2D(512, (3, 3), strides=2, padding='same') (x) 
    
    #x = Dense(1024) (x)
    x = Dense(256) (x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(1) (x)
    x = Activation('sigmoid')(x)
        
    model = Model(inputs=input, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def create_gan(generator, discriminator):
    discriminator.trainable=False
    gan_input = Input(shape=(32, 32, 3))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return gan

