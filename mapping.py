from __future__ import print_function
import tensorflow as tf
import csv
from tflearn.data_utils import load_csv, to_categorical
import tflearn
import numpy as np
from datetime import datetime
import math


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

        copy_data[i][0] = day_of_year
        copy_data[i][1] = sample[1]
        copy_data[i][2] = sample[2]
        copy_data[i][3] = sample[3]
        copy_data[i][4] = sample[4]
        copy_data[i][5] = sample[5]
        copy_data[i][6] = sample[6]
        copy_data[i][7] = sample[7]
        copy_data[i][8] = sample[8]
        copy_data[i][9] = sample[9]
        
    return copy_data

# def preprocess_labels(labels):
#      for i in range(len(labels)):
#          lab = float(labels[i])
#          if lab > 4500:
#              labels[i] = 9
#          elif lab > 4000:
#              labels[i] = 8
#          elif lab > 3500:
#              labels[i] = 7
#          elif lab > 3000:
#              labels[i] = 6
#          elif lab > 2500:
#              labels[i] = 5 
#          elif lab > 2000:
#              labels[i] = 4
#          elif lab > 1500:
#              labels[i] = 3
#          elif lab > 1000:
#              labels[i] = 2
#          elif lab > 500:
#              labels[i] = 1
#          else:
#              labels[i] = 0


def preprocess_labels(labels):
      for i in range(len(labels)):
          lab = float(labels[i])
          if lab > 4000:
              labels[i] = 4
          elif lab > 3000:
              labels[i] = 3
          elif lab > 2000:
              labels[i] = 2
          elif lab > 1000:
              labels[i] = 1
          else:
              labels[i] = 0


#  def preprocess_labels(labels):
#      for i in range(len(labels)):
#          lab = float(labels[i])
#          if lab > 2000:
#              labels[i] = 1
#          else:
#              labels[i] = 0


# def normalize(arr):
#     tmp_arr = np.zeros((len(arr),12)) 
#     for i in range(len(arr)):
#         for j in range(len(arr[0])):
#             tmp_arr[i][j] = (arr[i][j]-min(arr[:,j])) / (max(arr[:,j])-min(arr[:,j]))
#     return tmp_arr
            

# preprocess
training_data = preprocess_data(training_data)
# training_data = normalize(training_data)
preprocess_labels(training_labels)
training_labels = to_categorical(training_labels, 5)


# build neural network
net = tflearn.input_data(shape=[None,12])
net = tflearn.fully_connected(net,32)
net = tflearn.fully_connected(net,32)
net = tflearn.fully_connected(net,32)
# net = tflearn.fully_connected(net,32)
# net = tflearn.fully_connected(net,32)
net = tflearn.fully_connected(net, 5, activation='softmax')
net = tflearn.regression(net,learning_rate=0.001)

# define model
model = tflearn.DNN(net, clip_gradients=1.0, tensorboard_verbose=3, tensorboard_dir='./tmp/weather.log')
# start training (apply gradient descent algorithm)
model.fit(training_data, training_labels, n_epoch=25, batch_size=30, show_metric=True)
