import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.layers import Input, Dense, Add, Subtract
from tensorflow.keras.models import Model
#import numpy as np
#import matplotlib.pyplot as plt
#from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
#from mpl_toolkits.mplot3d import Axes3D

import tensorflow.keras as k



def MLP(act='tanh', nlayers='6', dim=3):

    model = Sequential()
    model.add(Dense(dim, activation='tanh', input_dim=dim))

    for i in range(nlayers-1):
        model.add(Dense(dim, activation='tanh'))

    model.add(Dense(1, activation='sigmoid'))

    return model

def ResNet(act='tanh', nlayers=51, epsilon=0.1, dim=2):

    input1 = Input(shape=(dim,))
    x1 = Dense(dim, activation=act)(input1)

    added = Add()([2*epsilon*x1, input1])

    for i in range(nlayers-1):
        x1 = Dense(dim, activation=act)(added)
        added = Add()([epsilon*x1, added])


    out = Dense(1, activation='sigmoid')(added)
    return Model(inputs=[input1], outputs=out)


def ResNetRegularized(act='tanh', nlayers=51, epsilon=0.001, dim=2):

    input1 = Input(shape=(dim,))
    x1 = Dense(dim, activation=act, kernel_regularizer=k.regularizers.l2(0.1))(input1)

    added = Add()([epsilon*x1, input1])

    for i in range(nlayers-1):
        x1 = Dense(dim, activation=act, kernel_regularizer=k.regularizers.l2(0.1/(i**2)))(added)
        added = Add()([epsilon*x1, added])


    out = Dense(1, activation='sigmoid')(added)
    return Model(inputs=[input1], outputs=out)


def ResNetRegularized(act='tanh', nlayers=51, epsilon=0.05, dim=2):

    input1 = Input(shape=(dim,))
    x1 = Dense(dim, activation=act, activity_regularizer=k.regularizers.l2(0.001))(input1)

    added = Add()([epsilon*x1, input1])

    for i in range(nlayers-1):
        x1 = Dense(dim, activation=act, activity_regularizer=k.regularizers.l2(0.001))(added)
        added = Add()([epsilon*x1, added])


    out = Dense(1, activation='sigmoid')(added)
    return Model(inputs=[input1], outputs=out)


def ResNetRegularized2(act='tanh', nlayers=51, nneurons=2, epsilon=0.05, momentum=0.2):

    input1 = Input(shape=(2,))

    residual = Dense(nneurons, activation=act, activity_regularizer=k.regularizers.l2(0.0001))(input1)

    v = Add()([epsilon*residual, input1])


    x1 = Add()([])

    for i in range(nlayers-1):
        x1 = Dense(nneurons, activation=act)(v)
        v = Add()([epsilon*x1, added])


    out = Dense(1, activation='sigmoid')(added)
    return Model(inputs=[input1], outputs=out)