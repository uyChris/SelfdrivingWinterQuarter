import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import time
from tensorflow.python.client import device_lib
from augmentation_functions import *

# print('printing list of devices')
# print(device_lib.list_local_devices())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# comment line above and uncomment line below for linux
# export TF_CPP_MIN_LOG_LEVEL = 2

"""
preprocess function:
- Cuts off top 80 pixels (as NVIDIA paper suggested)
- resizes image to (200, 66, 3) (as NVIDIA paper suggested)
- applies gaussian blur (as NVIDIA paper suggested)
- converts to YUV color space (as NVIDIA paper suggested)
"""

def preprocess(image):
    result = image[80:, :, :]
    result = cv2.resize(result, (200, 66), interpolation=cv2.INTER_AREA)
    result = cv2.GaussianBlur(result, (3, 3), 0)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
    return result

def generator(samples, batch_size=32):
    num_samples = len(samples)
    # print('len(images): ', len(images))
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            angles = []
            speeds = []
            # image_paths = []
            images = []

            for batch_sample in batch_samples:

                angle = float(batch_sample[0])/60.0
                angles.append(angle)
                speed = float(batch_sample[1])/60.0
                speeds.append(speed)

                image_path = batch_sample[2]
                image = Image.open(image_path)
                image = np.asarray(image)
                original_image = image
                image = preprocess(image)
                images.append(image)

                # augment image with 80% probability
                # for each batch_sample, we are appending two samples: the original sample and the augmented sample
                # so actual batch size is BATCH_SIZE * 2
                augmented_image, augmented_angle = augment_image(original_image, angle, p=0.8)
                angles.append(augmented_angle)
                speeds.append(speed)
                augmented_image = preprocess(augmented_image)
                images.append(augmented_image)



            images = np.asarray(images)
            # print('images.shape: ', images.shape)
            angles = np.asarray(angles)
            # print('angles.shape: ', angles.shape)
            images, angles = shuffle(images, angles)

            # input: an image as a numpy array
            # output(s): angle or angle and speed as numpy arrays
            # can also give initial weights, but this is optional
            # line below shows general format of the yield statement
            # yield (input , output, [initial weights])

            yield (images, angles) # for yielding one output
            # yield (images, np.transpose(np.asarray([angles,speeds]))) # for yielding two or more outputs: just keep expanding on this list and make sure to change the number of output nodes in models.py


if __name__ == '__main__':
    file_name = 'data.p'

    with open(file_name, 'rb') as f:
        samples = pickle.load(f)
    print()
    print('loading training examples from pickle file')
    print('number of training examples: ', len(samples))

    # MODEL = "modified_lenet"
    MODEL = "modified_lenet"  # can be "modified_lenet", "nvidia"

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print('len(train_samples): ', len(train_samples))
    print('len(validation_samples): ', len(validation_samples))

    if (MODEL == "modified_lenet"):
        TRIAL_NUMBER = 5  # which sub-folder to save trained models to
        BATCH_SIZE = 32 # note: if we are augmenting images, then batch_size is actually twice this number
        EPOCHS = 1 # number of iterations to train for
        LR_FACTOR = 0.1  # factor by which to decrease learning rate if training plateaus

        train_generator = generator(train_samples, batch_size=BATCH_SIZE)
        validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

        # reduce learning rate if model plateaus
        # factor defines the factor by which to reduce learning rate
        # patience defines how many epochs to wait during a plateau before reducing learning rate
        reduce_lr = ReduceLROnPlateau(factor=LR_FACTOR, patience=2, epsilon=0.00001, verbose=1)

        model_save_path = ("models/modified_lenet/%d/weights_only_model_modified_lenet_lrFactor=%.3f_batchSize=%d_epoch={epoch:02d}_valLoss={val_loss:.6f}.h5" % (TRIAL_NUMBER, LR_FACTOR, BATCH_SIZE))
        modelCheckpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_loss', verbose=0, period=1, save_weights_only=1)  # save model after every epoch

        modified_lenet_model = load_model('blank_modified_lenet_model.h5')
        modified_lenet_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        modified_lenet_model.summary()

        history = modified_lenet_model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/BATCH_SIZE, callbacks=[reduce_lr, modelCheckpoint], epochs=EPOCHS, validation_data=validation_generator, validation_steps=len(validation_samples)/BATCH_SIZE)

        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()


    elif (MODEL == "nvidia"):
        TRIAL_NUMBER = 1  # which sub-folder to save trained models to
        BATCH_SIZE = 32
        EPOCHS = 10
        LR_FACTOR = 0.1  # factor by which to decrease learning rate if training plateaus

        train_generator = generator(train_samples, batch_size=BATCH_SIZE)
        validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

        # reduce learning rate if model plateaus
        # factor defines the factor by which to reduce learning rate
        # patience defines how many epochs to wait during a plateau before reducing learning rate
        reduce_lr = ReduceLROnPlateau(factor=LR_FACTOR, patience=2, epsilon=0.00001, verbose=1)

        model_save_path = ("models/nvidia/%d/weights_only_model_nvidia_lrFactor=%.3f_batchSize=%d_epoch={epoch:02d}_valLoss={val_loss:.6f}.h5" % (TRIAL_NUMBER, LR_FACTOR, BATCH_SIZE))
        modelCheckpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_loss', verbose=0, period=1, save_weights_only=1)  # save model weights after every epoch

        nvidia_model = load_model('blank_nvidia_model.h5')
        nvidia_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        nvidia_model.summary()

        history = nvidia_model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/BATCH_SIZE, callbacks=[reduce_lr, modelCheckpoint], epochs=EPOCHS, validation_data=validation_generator, validation_steps=len(validation_samples)/BATCH_SIZE)

        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()


    print('done')
