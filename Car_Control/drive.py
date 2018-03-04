import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import os

import time

import RPi.GPIO as GPIO
# import readchar
import pigpio

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# comment line above and uncomment line below for linux
# export TF_CPP_MIN_LOG_LEVEL = 2


# model_path = "models/modified_lenet/1/weights_only_model_modified_lenet_lrFactor=0.100_batchSize=32_epoch=09_valLoss=0.019591.h5"
model_path = "models/modified_lenet/4/weights_only_model_modified_lenet_lrFactor=0.100_batchSize=64_epoch=0090_valLoss=0.003751.h5"


# model_path = "models/nvidia/model_nvidia_adam_optimizer_epochs=10_trainloss=0.0523_valloss=0.0290.h5"
# model_path = "models/nvidia/1/model_nvidia_lrFactor=0.100_batchSize=32_epoch=09_valLoss=0.028470.h5"
# model_path = "models/modified_lenet/3/model_modified_lenet_lrFactor=0.100_batchSize=32_epoch=0060_valLoss=0.011601.h5"

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
modified_lenet_model.add(Dense(1))

modified_lenet_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
modified_lenet_model.summary()

modified_lenet_model.load_weights(model_path)
model = modified_lenet_model

# Define pins used.
servoPin = 24
pi = pigpio.pi()


PIN = 18
PWMA1 = 6
PWMA2 = 13
PWMB1 = 20
PWMB2 = 21
PWMC1 = 24
D1 = 12
D2 = 26

PWM = 50

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(PIN,GPIO.IN,GPIO.PUD_UP)
GPIO.setup(PWMA1,GPIO.OUT)
GPIO.setup(PWMA2,GPIO.OUT)
GPIO.setup(PWMB1,GPIO.OUT)
GPIO.setup(PWMB2,GPIO.OUT)
GPIO.setup(PWMC1,GPIO.OUT)
GPIO.setup(D1,GPIO.OUT)
GPIO.setup(D2,GPIO.OUT)
p1 = GPIO.PWM(D1,500)
p2 = GPIO.PWM(D2,500)
p3 = GPIO.PWM(PWMC1,50)
p1.start(0)
p2.start(0)

# This function will take a float from -1 to 1 and map it to a degree from 0 to 180


def floattodeg (num):
    # Only allow angle to go from 30 to 150
    # Chassis is blocking wheels from turning anymore
    return (num * 60) + 90


# Expects an angle between -60 and 60


def setAngle(angle):
    if (angle < -60):
        angle = -60
    elif (angle > 60):
        angle = 60
    angle = angle + 90
    #print("Angle: ", angle)
    pulseWidth = angle*5.55555555555555555555555555555555555556 + 1000
    #print("pulseWidth: ", pulseWidth)
    pi.set_servo_pulsewidth(servoPin,pulseWidth)
    time.sleep(0.001)



#Sets motor output based on four input values through outputting to GPIO pins.


def	set_motor(A1, A2, B1, B2):
	GPIO.output(PWMA1, A1)
	GPIO.output(PWMA2, A2)
	GPIO.output(PWMB1, B1)
	GPIO.output(PWMB2, B2)

# Forward involves pushing both motors forward.


def forward():
    # PWMA1 and PWMB1 set high.
    set_motor(1, 0, 1, 0)


def stop():
	set_motor(0,0,0,0)

def reverse():
    #PWMA2 and PWMB2 set high.
	set_motor(0,1,0,1)

def left():
    #PWMA1 and PWMB2 set high.
	set_motor(1,0,0,0)

def right():
    #PWMA2 and PWMB1 set high.
	set_motor(0,0,1,0)




def preprocess(image):
    result = image[80:, :, :]
    result = cv2.resize(result, (200, 66), interpolation=cv2.INTER_AREA)
    result = cv2.GaussianBlur(result, (3, 3), 0)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
    return result


"""
# for displaying actual vs. predicted angles on our own data
file_name = 'data.p'

with open(file_name, 'rb') as f:
    angles, speeds, images = pickle.load(f)

print('angles.shape: ', angles.shape)
print('speeds.shape: ', speeds.shape)
print('images.shape: ', images.shape)


font = cv2.FONT_HERSHEY_SIMPLEX
fig, axs = plt.subplots(2, 5, figsize=(12, 5))
axs = axs.ravel()
count = 0
for i in range(0, 10000, 1000):
    image = images[i]
    image = preprocess(image)
    image_copy = np.copy(image)
    actual_angle = angles[i]
    predicted_angle = float(model.predict(image[None, :, :, :], batch_size=1))*60.0
    predicted_angle = round(predicted_angle, 6)
    cv2.putText(image_copy, "Actual: %s" % (str(actual_angle)), (10, 20), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image_copy, "Predicted: %s" % (str(predicted_angle)), (10, 50), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    axs[count].imshow(image_copy)
    count += 1
plt.show()
"""

cap = cv2.VideoCapture(0)

# Make Car go forward at constant speed
reverse()
PWM = 10 # Defines the speed (ranges from 0 to 100)
p1.ChangeDutyCycle(PWM)
p2.ChangeDutyCycle(PWM)
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    # Capture frame-by-frame
    t1 = time.time()
    ret, frame = cap.read()
    if (ret==True):
    
        preprocessed_frame = preprocess(frame)
        predicted_angle = float(model.predict(preprocessed_frame[None, :, :, :], batch_size=1))*60.0
        t2 = time.time()
        delta = t2-t1
        setAngle(predicted_angle)
        # print("delta: ", delta)
        # print("predicted_angle: ", predicted_angle)
        print()
        # Display the resulting frame
        frame_copy = np.copy(frame)
        cv2.putText(frame_copy, "Angle: %s" % (str(predicted_angle)), (10, 20), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_copy, "Time: %s" % (str(delta)), (10, 40), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        p1.ChangeDutyCycle(0)
        p2.ChangeDutyCycle(0)
        stop()
        break

# When everything done, release the capture
forward()
p1.ChangeDutyCycle(0)
p2.ChangeDutyCycle(0)
stop()
cap.release()
cv2.destroyAllWindows()

print('done')
