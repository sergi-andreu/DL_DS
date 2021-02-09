from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from mpl_toolkits.mplot3d import Axes3D

act='tanh'
nlayers=51
nneurons1=2
nneurons2=2

def 2D_ResNet():

    input1 = Input(shape=(2,))
    x1 = Dense(nneurons1, activation=act)(input1)

    added = Add()([0.05*x1, input1])

    for i in range(nlayers-1):
        x1 = Dense(nneurons, activation=act)(added)
        added = Add()([0.05*x1, added])


    out = Dense(1, activation='sigmoid')(added)
    return Model(inputs=[input1], outputs=out)