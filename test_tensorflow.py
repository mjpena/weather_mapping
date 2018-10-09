from __future__ import print_function
import csv
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import tflearn
from datetime import datetime
from tflearn.datasets import titanic
from tflearn.data_utils import load_csv, to_categorical


# The file containing the weather samples (including the column header)
WEATHER_SAMPLE_FILE = 'weather_dataset.csv'
data, labels = load_csv(WEATHER_SAMPLE_FILE, target_column=11, columns_to_ignore=[0])

TrainingSetFeatures = data
TrainingSetLabels = labels

def preprocessor(data):
	copyData = np.zeros((len(data), 12))
	for i in range(len(data)):
		sample = data[i]
		#grab the date element
		dayStr = sample[0]
		dayOfYear = datetime.strptime(dayStr, "%m/%d/%Y").timetuple().tm_yday
		hours = int(sample[1])
		hourVectorReal = math.cos(2*math.pi * (hours/24))
		hourVectorImg = math.sin(2*math.pi * (hours/24))		
		dayVectorReal = math.cos(2*math.pi * (dayOfYear/365))
		dayVectorImg = math.sin(2*math.pi * (dayOfYear/365))
		copyData[i][0] = hourVectorReal 
		copyData[i][1] = hourVectorImg 
		copyData[i][2] = dayVectorReal 
		copyData[i][3] = dayVectorImg
		copyData[i][4] = sample[2]
		copyData[i][5] = sample[3]
		copyData[i][6] = sample[4]
		copyData[i][7] = sample[5]
		copyData[i][8] = sample[6]
		copyData[i][9] = sample[7]
		copyData[i][10] = sample[8]
		copyData[i][11] = sample[9]
	return copyData

def categorizeLabels(labels):
	for i in range(len(labels)):
		evSample = float(labels[i])
		if evSample > 4000:
			labels[i] = 4
		elif evSample > 3000:
			labels[i] = 3
		elif evSample > 2000:
			labels[i] = 2
		elif evSample > 1000:
			labels[i] = 1
		else:
			labels[i] = 0

TrainingSetFeatures = preprocessor(TrainingSetFeatures)
categorizeLabels(TrainingSetLabels)
TrainingSetLabels = to_categorical(TrainingSetLabels, 5)

#create a test set from the number of samples and traning set
net = tflearn.input_data(shape=[None, 12])
net = tflearn.fully_connected(net, 64)
net = tflearn.fully_connected(net, 64)
net = tflearn.fully_connected(net, 64)
net = tflearn.fully_connected(net, 5, activation="softmax")
net = tflearn.regression(net)
# categorized the data into bins for and that should be the number of 0.88888

# Define model
model = tflearn.DNN(net)

# Start training (apply gradient descent algorithm)
model.fit(TrainingSetFeatures, TrainingSetLabels, n_epoch=2, batch_size=12, show_metric=True, validation_set=0.1)

# Let's create some data for DiCaprio and Winslet
lowOutput =  [0, 0, 0, 0, 0, 9.92, 0.37, -0.01, 89.12, 4.72, 29.19, 29.98]
highOutput = [0, 0, 0, 0, 0, 10, 6.16, 1.26, 68.96, 0, 29.26, 30.05]

pred = model.predict([lowOutput, highOutput])

# find index
lowOutputPrediction = pred[0].argmax()
highOutputPrediction = pred[1].argmax()

print("lowoutput estimate:", lowOutputPrediction)
print("highOutput estimate:", highOutputPrediction)