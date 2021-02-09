import distributions
import models

import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from mpl_toolkits.mplot3d import Axes3D

from tqdm.keras import TqdmCallback

import matplotlib.pyplot as plt


dim = 3

xy, z = distributions.generatorringsgap(1000, dim=dim)


if dim==3:

    thirddimension = np.zeros((np.shape(xy[0])[0]))

    xy = np.stack([xy[0], xy[1], thirddimension])


np.save("Inputs", xy)
np.save("Labels", z)

#model = models.ResNet(nlayers=51, dim=dim, epsilon=0.05)
model = models.MLP(nlayers=10, dim=3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.transpose(xy), z, test_size=0.8, random_state=0)

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=5000, batch_size=50, verbose=0)

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=5, batch_size=50, verbose=1)

model.save("trained_model")

if __name__ == "__main__":

    """

    fig, ax = plt.subplots()

    ax.set_aspect('equal', 'box')
    ax.tick_params(direction="in")
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])

    ax.scatter(np.transpose(X_train)[0], np.transpose(X_train)[1], c=distributions.binarycolor(y_train, "b", "y"))
    fig.tight_layout()
    plt.show()

    """

    """

    fig, ax = plt.subplots()

    ax.set_aspect('equal', 'box')
    ax.tick_params(direction="in")
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_xticks([-5,0,5])
    ax.set_yticks([-5,0,5])

    #ax.grid(False)

    slope=3
    offset=0
    #offset=2
    iii=199

    jjj=000

    for iii in range(51):

        #activation_model = models.Model(inputs=model.input, outputs=model.layers[j].output)
        activation_model = models.Model(inputs=model.input, outputs=model.layers[slope*iii+offset].output)
        activations = activation_model.predict(np.transpose(xy))
        ax.scatter(np.transpose(activations)[0], np.transpose(activations)[1], c=distributions.binarycolor(z, "b", "y"))

        grid_x = np.linspace(-1,1,10)
        grid_y = np.linspace(-1,1,10)
        grid_z = np.zeros((10,))


        for i in range(10):

            vector = []

            for j in range(10):
                vector.append([grid_x[i], grid_y[j]])

            act = activation_model.predict(np.array(vector))                  
            ax.plot(np.transpose(act)[0], np.transpose(act)[1], c="k")

        for i in range(10):

            vector = []

            for j in range(10):
                vector.append([grid_x[j], grid_y[i]])

            act = activation_model.predict(np.array(vector))                  
            ax.plot(np.transpose(act)[0], np.transpose(act)[1],  c="k")

        ax.set_axis_off()
        #ax.view_init(-30, 0)

        plt.title("Hidden layer "+str(iii))

        fig.savefig("2D_ResNet_Ring/im"+str(iii).zfill(2)+".png",bbox_inches='tight')

        #plt.show()
        ax.clear()
        #plt.close("all")

    """