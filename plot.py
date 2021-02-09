import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import matplotlib.pyplot as plt

import distributions

from tensorflow.keras import models

model = tf.keras.models.load_model('trained_model')

xy = np.load("Inputs.npy")
z = np.load("Labels.npy")

CARTESIANGRID = False
AXISOFF = True

dim = 3

size = 3
linewidth = 0.5

grid_x = np.linspace(-1,1,10)
grid_y = np.linspace(-1,1,10)
grid_z = np.zeros((10,))

colors = distributions.binarycolor(z, "b", "y")

#ax.grid(False)

#slope=3
slope=1
offset=0
#offset=2

nlayers=10

azimutal = 20
elevation = 30


activation_model = models.Model(inputs=model.input, outputs=model.layers[slope*(nlayers-1)+offset].output)
activations = activation_model.predict(np.transpose(xy))

max_value = tf.math.reduce_max(activations)
min_value = tf.math.reduce_min(activations)
max_value = max_value.numpy()
min_value = min_value.numpy()
#max_value = np.ceil(max_value)
#min_value = np.floor(min_value)
max_value = 0.1 + max_value
min_value = -0.1 + min_value


if dim==2:
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')

if dim==3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


#ax.tick_params(direction="in")

ax.set_xlim([min_value,max_value])
ax.set_ylim([min_value,max_value])

if dim==3:
    ax.set_zlim([min_value, max_value])

ax.set_xticks([])
ax.set_yticks([])

if dim==3:
    ax.set_zticks([])

if dim==2:
    ax.scatter(xy[0], xy[1], c=colors, s=size)


elif dim==3:
    ax.scatter(xy[0], xy[1], xy[2], c=colors, s=size)
    ax.view_init(azim=azimutal, elev=elevation)



if CARTESIANGRID:

    if dim==2:

        for i in range(10):
            vector = []

            for j in range(10):
                vector.append([grid_x[i], grid_y[j]])

            act = np.transpose(np.array(vector))         
            ax.plot(act[0], act[1], c="k", linewidth=linewidth)

        for i in range(10):

            vector = []

            for j in range(10):
                vector.append([grid_x[j], grid_y[i]])

            act = np.transpose(np.array(vector))           
            ax.plot(act[0], act[1],  c="k", linewidth=linewidth)

    if dim==3:
            for i in range(10):
                vector = []

                for j in range(10):
                    vector.append([grid_x[i], grid_y[j], 0])

                act = np.transpose(np.array(vector))                
                ax.plot(act[0], act[1], act[2], c="k", linewidth=linewidth)

            for i in range(10):

                vector = []

                for j in range(10):
                    vector.append([grid_x[j], grid_y[i], 0])

                act = np.transpose(np.array(vector))               
                ax.plot(act[0], act[1], act[2],  c="k", linewidth=linewidth)

if AXISOFF:
    ax.set_axis_off()
    ax.grid(False)
#ax.view_init(-30, 0)

plt.title("Hidden layer 0")

fig.savefig("2D_ResNet_Ring/im00.png",bbox_inches='tight')

#plt.show()
ax.clear()


for iii in range(1,nlayers):

    if dim==2:
        ax.set_aspect('equal', 'box')

    if dim==3:
        ax.view_init(azim=azimutal, elev=elevation)

    ax.tick_params(direction="in")
    ax.set_xlim([min_value,max_value])
    ax.set_ylim([min_value,max_value])
    


    ax.set_xticks([])
    ax.set_yticks([])

    activation_model = models.Model(inputs=model.input, outputs=model.layers[slope*iii+offset].output)
    activations = activation_model.predict(np.transpose(xy))

    if dim==2:
        ax.scatter(np.transpose(activations)[0], np.transpose(activations)[1], c=colors, s=size)

    if dim==3:
        ax.scatter(np.transpose(activations)[0], np.transpose(activations)[1], np.transpose(activations)[2], c=colors, s=size)

    if CARTESIANGRID:

        if dim==2:
            for i in range(10):
                vector = []

                for j in range(10):
                    vector.append([grid_x[i], grid_y[j]])

                act = np.transpose(activation_model.predict(np.array(vector)))                
                ax.plot(act[0], act[1], c="k", linewidth=linewidth)

            for i in range(10):

                vector = []

                for j in range(10):
                    vector.append([grid_x[j], grid_y[i]])

                act = np.transpose(activation_model.predict(np.array(vector)))              
                ax.plot(act[0], act[1],  c="k", linewidth=linewidth)

        if dim==3:
            for i in range(10):
                vector = []

                for j in range(10):
                    vector.append([grid_x[i], grid_y[j], 0])

                act = np.transpose(activation_model.predict(np.array(vector)))
               
                ax.plot(act[0], act[1], act[2], c="k", linewidth=linewidth)

            for i in range(10):

                vector = []

                for j in range(10):
                    vector.append([grid_x[j], grid_y[i], 0])

                act = np.transpose(activation_model.predict(np.array(vector)))

                ax.plot(act[0], act[1], act[2],  c="k", linewidth=linewidth)


    if AXISOFF:
        ax.set_axis_off()
    #ax.view_init(-30, 0)

    plt.title("Hidden layer "+str(iii))

    fig.savefig("2D_ResNet_Ring/im"+str(iii).zfill(2)+".png",bbox_inches='tight')

    #plt.show()
    ax.clear()
    #plt.close("all")