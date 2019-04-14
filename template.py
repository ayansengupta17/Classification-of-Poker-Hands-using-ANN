import pandas as pd
import numpy as np
from collections import Counter
import preprocessing as prep

############### For 2 hidden layer use model
############### For 3 hidden layer  use model3
############### For 4 hidden layer use model 4

import model
#import model3 as model
#import model4 as model
import csv

#Import training Data from train.csv

train_data_raw = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data_raw = train_data_raw[0:200000]

### preprocessing ###

#separation of labels and trainind data
labels_temp, train_data = prep.get_labels(train_data_raw)

train_data_temp = np.array(train_data)
test_data_temp = np.array(test_data)

#encoding of training data and labels
train_data = prep.encode_train(train_data_temp)
test_data = prep.encode_train(test_data_temp)

train_data = np.array(train_data)
test_data = np.array(test_data)


label_encoder = prep.encode_class(labels_temp[0], 10)
labels = np.array(label_encoder)

# k is th efold for k cross validation
k=5
y_list = []
w_list = []
acc_list = []

for training, validation, training_label, validation_label in model.k_fold_cross_validation(train_data, labels, k):
	
	
	w = model.initialize_w2(train_data)
	weights = model.stochastic_model(train_data= training, output=training_label, step=0.01,mf=0.4, iterations=200, lambd=1, weights=w)
	y, acc = model.accuracy2(weights, (validation), validation_label)
	y_list.append(y)
	w_list.append(weights)
	acc_list.append(acc)

w_best = w_list[list(acc_list).index(max(acc_list))]
y, acc = model.accuracy2(w_best, (validation), validation_label)

model.predict(w_best,test_data,labels, "output99.csv")

