import csv
import numpy as np
from PIL import Image
import pickle
import cv2
import random


def preprocess(image):
    result = image[80:, :, :]
    result = cv2.resize(result, (200, 66), interpolation=cv2.INTER_AREA)
    result = cv2.GaussianBlur(result, (3, 3), 0)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
    return result

# list of all data text files
paths = ["output_logs/output_log2/output_log2.txt", "output_logs/output_log3/output_log3.txt"]

# for every data text file
for index,path in enumerate(paths):
    # read in all the samples in that text file and append to list
    # samples is a list of lists
    # each sub-list contains 3 elements: angle, speed, image path
    # Example of one sub-list/training example: ['0', '0', 'output_logs/output_log2/frames/frame1780.jpg']
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        if (index == 0):
            samples = list(csv_reader)
        else:
            samples += list(csv_reader)

print('number of training samples: ', len(samples)) # print out number of training examples
index = random.randint(0,len(samples))
print('a random training example: ', samples[index]) # examine one training example

# save training examples to pickle file
print()
print('saving training examples to pickle file')
file_name = 'data.p'
with open(file_name, 'wb') as f:
    # pickle.dump([angles, speeds, images], f)
    pickle.dump(samples, f)

# make sure we can load data from pickle file
with open(file_name, 'rb') as f:
    samples = pickle.load(f)
print()
print('loading training examples from pickle file')
print('number of training examples: ', len(samples))
print('type(samples): ', type(samples))
print(samples[index])
