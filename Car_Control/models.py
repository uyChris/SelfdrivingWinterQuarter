import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import time
from tensorflow.python.client import device_lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# comment line above and uncomment line below for linux
# export TF_CPP_MIN_LOG_LEVEL = 2

# Define the input shape
input_shape = (66, 200, 3)

# Define modified lenet model
modified_lenet_model = Sequential()
print()
print('Modified Lenet Model:\n')
# pixel_normalized_and_mean_centered = pixel / 255 - 0.5
modified_lenet_model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
print(modified_lenet_model.output_shape)
# modified_lenet_model.add(Convolution2D(nb_filter=6, nb_row=5, nb_col=5,
#                        activation='relu', border_mode='valid',
#                        subsample=(1, 1), dim_ordering='tf'))
modified_lenet_model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', data_format='channels_last'))
print(modified_lenet_model.output_shape)
modified_lenet_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

print(modified_lenet_model.output_shape)
# modified_lenet_model.add(Convolution2D(nb_filter=16, nb_row=5, nb_col=5,
#                        activation='relu', border_mode='valid',
#                        subsample=(1, 1)))
modified_lenet_model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', data_format='channels_last'))

print(modified_lenet_model.output_shape)
# modified_lenet_model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5,
#                        activation='relu', border_mode='valid',
#                        subsample=(1, 1)))
modified_lenet_model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', data_format='channels_last'))
print(modified_lenet_model.output_shape)
modified_lenet_model.add(Flatten())
print(modified_lenet_model.output_shape)
modified_lenet_model.add(Dense(120, activation='relu'))
modified_lenet_model.add(Dropout(0.25))
print(modified_lenet_model.output_shape)
modified_lenet_model.add(Dense(84, activation='relu'))
modified_lenet_model.add(Dropout(0.25))
print(modified_lenet_model.output_shape)
modified_lenet_model.add(Dense(2))

modified_lenet_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
modified_lenet_model.summary()
print()

# Define nvidia model
nvidia_model = Sequential()
print('Nvidia Model:\n')
# pixel_normalized_and_mean_centered = pixel / 255 - 0.5
nvidia_model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
print(nvidia_model.output_shape)

nvidia_model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu', data_format='channels_last'))
print(nvidia_model.output_shape)

nvidia_model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu', data_format='channels_last'))
print(nvidia_model.output_shape)

nvidia_model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu', data_format='channels_last'))
print(nvidia_model.output_shape)

nvidia_model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', data_format='channels_last'))
print(nvidia_model.output_shape)

nvidia_model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', data_format='channels_last'))
print(nvidia_model.output_shape)

nvidia_model.add(Flatten())
print(nvidia_model.output_shape)
nvidia_model.add(Dropout(0.25))

nvidia_model.add(Dense(100, activation='relu'))
print(nvidia_model.output_shape)
nvidia_model.add(Dropout(0.25))

nvidia_model.add(Dense(50, activation='relu'))
print(nvidia_model.output_shape)
nvidia_model.add(Dropout(0.25))

nvidia_model.add(Dense(10, activation='relu'))
print(nvidia_model.output_shape)
nvidia_model.add(Dropout(0.25))

nvidia_model.add(Dense(1))

nvidia_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
nvidia_model.summary()
print()

modified_lenet_model.save('blank_modified_lenet_model.h5')
nvidia_model.save('blank_nvidia_model.h5')

print('models saved')
