from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Input, Activation, Lambda, BatchNormalization, Add
from helpfunc import PS

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

    x = Conv2D(64, (9, 9), activation='relu', padding='same') (input)
    skipRes = x

    # Resnet layers
    for i in range(10):
        x = resBlock(x, 64)

    x = Conv2D(64, (3, 3), activation='relu', padding='same') (x) 
    x = BatchNormalization()(x)
    x = Add()([skipRes, x])

    # Sub-pixel convolution layer 1
    r = 8 #Upscale x4
    x = Conv2D(3*r*r, (3, 3), activation='relu', padding='same') (x) 
    x = Lambda(lambda x: PS(x, r))(x)
    x = Activation('tanh')(x)

    # x = Conv2D(128, (3, 3), activation='relu', padding='same') (x) 
    #Sub-pixel convolution layer 2
    # r = 2 #Upscale x2
    # x = Conv2D(3*r*r, (3, 3), activation='relu', padding='same') (x) 
    # x = Lambda(lambda x: PS(x, r))(x)
    # x = Activation('tanh')(x)

    x = Conv2D(3, (9, 9), activation='relu', padding='same') (x) 
    return x

