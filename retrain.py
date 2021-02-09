from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


xy = np.load("Inputs.npy")
z = np.load("Labels.npy")

X_train, X_test, y_train, y_test = train_test_split(np.transpose(xy), z, test_size=0.5, random_state=0)

model = tf.keras.models.load_model('trained_model')


model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=2000, batch_size=10, verbose=1)

model.save("trained_model")

