from __future__ import print_function
import tensorflow as tf
import csv
from tflearn.data_utils import load_csv, to_categorical
import pandas as pd
import tflearn
import numpy as np
from datetime import datetime
import math
import matplotlib.pyplot as plt


# tflearn tutorial from tflearn.org/tutorials/quickstart.html
# loading csv file; target_column is solar energy; ignore random column and first dates column
WEATHER_SAMPLE_FILE = 'weather_dataset.csv'
data, labels = load_csv(WEATHER_SAMPLE_FILE, target_column=11, columns_to_ignore=[0])

training_data = data
training_labels = labels

# preprocessing for dates column; [i][1] gets all the dates
def preprocess_data(data):
    copy_data = np.zeros((len(data),12))     #creates empty array of zeroes
    for i in range(len(data)):
        sample = data[i]
        day_str = sample[0]
        day_of_year = datetime.strptime(day_str, "%m/%d/%Y").timetuple().tm_yday
        hours = int(sample[1])    
        hour_real = math.cos(2*math.pi * (hours/24))
        hour_img = math.sin(2*math.pi * (hours/24))
        day_real = math.cos(2*math.pi * (day_of_year/365))
        day_img = math.sin(2*math.pi * (day_of_year/365))
        
        copy_data[i][0] = hour_real
        copy_data[i][1] = hour_img
        copy_data[i][2] = day_real
        copy_data[i][3] = day_img
        copy_data[i][4] = sample[2]
        copy_data[i][5] = sample[3]
        copy_data[i][6] = sample[4]
        copy_data[i][7] = sample[5]
        copy_data[i][8] = sample[6]
        copy_data[i][9] = sample[7]
        copy_data[i][10] = sample[8]
        copy_data[i][11] = sample[9]
    return copy_data

def preprocess_labels(labels):
     for i in range(len(labels)):
         lab = float(labels[i])
         if lab > 4500:
             labels[i] = 9
         elif lab > 4000:
             labels[i] = 8
         elif lab > 3500:
             labels[i] = 7
         elif lab > 3000:
             labels[i] = 6
         elif lab > 2500:
             labels[i] = 5 
         elif lab > 2000:
             labels[i] = 4
         elif lab > 1500:
             labels[i] = 3
         elif lab > 1000:
             labels[i] = 2
         elif lab > 500:
             labels[i] = 1
         else:
             labels[i] = 0

# def preprocess_labels(labels):
#     for i in range(len(labels)):
#         lab = float(labels[i])
#         if lab > 4000:
#             labels[i] = 4
#         elif lab > 3000:
#             labels[i] = 3
#         elif lab > 2000:
#             labels[i] = 2
#         elif lab > 1000:
#             labels[i] = 1
#         else:
#             labels[i] = 0


# preprocess
training_data = preprocess_data(training_data)
preprocess_labels(training_labels)
training_labels = to_categorical(training_labels, 10)


# build neural network
net = tflearn.input_data(shape=[None,12])
net = tflearn.fully_connected(net,64)
net = tflearn.fully_connected(net,64)
net = tflearn.fully_connected(net,64)
net = tflearn.fully_connected(net,64)
net = tflearn.fully_connected(net,64)
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net)

# define model
model = tflearn.DNN(net)
# start training (apply gradient descent algorithm)
model.fit(training_data, training_labels, n_epoch=2, batch_size=12, show_metric=True, validation_set=0.1)

low_output =  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mid_output = [0, 290, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0]
high_output = [0, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

pred = model.predict(training_data[0:1000])
plt.plot(pred, label='Predicted')
plt.plot(labels[0:1000], label='Actual')
plt.title("Predicted vs. Actual Solar Power")
plt.ylabel("Solar power")
plt.xlabel("Time")
plt.legend()
plt.show()

# find index
low_output_prediction = pred[0].argmax()
# mid_output_prediction = pred[1].argmax()
high_output_prediction = pred[2].argmax()

print("lowoutput estimate:", low_output_prediction)
# print("midoutput estimate:", mid_output_prediction)
print("highoutput estimate:", high_output_prediction)


