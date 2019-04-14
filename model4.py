import numpy as np
import csv

#Activation functions

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def sigmoid_backward(x):
    return (sigmoid(x)*(1.0-(sigmoid(x))))


def softmax_backward(x):
    return softmax(x)*(1.0-softmax(x))


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def tanh_backward(z):
    return 1 - tanh(z)**2


def relu_backward(x):
     x[x<=0] = 0
     x[x>0] = 1
     return x


def relu(data, epsilon=0.01):
    return np.maximum(epsilon * data, data)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1) 

def softmax2(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def backward(x):
    return (x)*(1.0-(x))



def initialize_w2(train_data):
	''' Initialization of weights are done by this function. 18 neurons in the first hidden layer
		and 10 neurons in the second hidden layer has been considered.
	'''
	w1 = 2*np.random.random((train_data.shape[1], 18 )) -1
	w2 = 2*np.random.random((18,18)) -1
	w3 = 2*np.random.random((18,18)) -1
	w4 = 2*np.random.random((18,10)) -1
	w5 = 2*np.random.random((10, 10)) -1
	weights = [w1, w2, w3, w4, w5]
	return weights




def stochastic_model(train_data, output, step, mf, iterations, lambd, weights):
	'''
	This is the stochastic model of our neural network. Input layer has 85 neurons. Two hidden layers has been considered for this particular 
	funtion.

	train_data : training data for the model
	output     : labels for the supervised learning
	step       : step size for the weight updates
	mf         : momentum factor
	iterations : number of loops over the whole data set.
	lambd      : lambda for regularization
	weights    : initial weights for the model


	'''
	output = np.asarray(output)
	prev_dc_dw5 = 0
	prev_dc_dw4 = 0
	prev_dc_dw3 = 0
	prev_dc_dw2 = 0
	prev_dc_dw1 = 0
	p_cost = 1

	w1 = weights[0]
	w2 = weights[1]
	w3 = weights[2]
	w4 = weights[3]
	w5 = weights[4]

	for j in range(iterations):
		
		print("epoch= {0}".format(j),)
		for k in range(len(train_data)):

			# forward propagation

			l_0 = np.array([train_data[k]])
			l_1 = sigmoid(np.dot(l_0,w1))
			l_2 = sigmoid(np.dot(l_1,w2))
			l_3 = sigmoid((np.dot(l_2,w3)))
			l_4 = sigmoid((np.dot(l_3,w4)))
			l_5 = sigmoid((np.dot(l_4,w5)))
			y3 = l_5

			#back propagation

			l_5_error = np.array([output[k]]) - l_5
			l_5_delta = l_5_error * backward(l_5)
			l_4_error = l_5_delta.dot(w5.T)
			l_4_delta = l_4_error * backward(l_4)
			l_3_error = l_4_delta.dot(w4.T)
			l_3_delta = l_3_error * backward(l_3)
			l_2_error = l_3_delta.dot(w3.T)
			l_2_delta = l_2_error * backward(l_2)
			l_1_error = l_2_delta.dot(w2.T)
			l_1_delta = l_1_error * backward(l_1)


			dc_dw5 = l_4.T.dot(l_5_delta) + lambd/(2*len(train_data))*w5
			dc_dw4 = l_3.T.dot(l_4_delta) + lambd/(2*len(train_data))*w4
			dc_dw3 = l_2.T.dot(l_3_delta) + lambd/(2*len(train_data))*w3
			dc_dw2 = l_1.T.dot(l_2_delta) + lambd/(2*len(train_data))*w2
			dc_dw1 = l_0.T.dot(l_1_delta) + lambd/(2*len(train_data))*w1

			# weight updates with momentum

			w5 += step*dc_dw5 + prev_dc_dw5
			w4 += step*dc_dw4 + prev_dc_dw4
			w3 += step*dc_dw3 + prev_dc_dw3
			w2 += step*dc_dw2 + prev_dc_dw2
			w1 += step*dc_dw1 + prev_dc_dw1

			prev_dc_dw5 = l_4.T.dot(l_5_delta) 
			prev_dc_dw4 = l_3.T.dot(l_4_delta) 
			prev_dc_dw3 = l_2.T.dot(l_3_delta)  
			prev_dc_dw2 = l_1.T.dot(l_2_delta)  
			prev_dc_dw1 = l_0.T.dot(l_1_delta)  

		#cross entropy loss function

		cost = (-1 / len(train_data)) * np.sum(output.T * np.log(y3.T) + (1 - output.T) * (np.log(1 - y3.T))) + lambd/(2*len(train_data)) * (np.sum(np.square(weights[0])) + np.sum(np.square(weights[1])) + np.sum(np.square(weights[2])))
		

	return [w1, w2, w3, w4, w5]



def k_fold_cross_validation(items, labels, k):
	'''function for k fold cross validation.

		items: input data set
		labels: input labels
		k : size of fold

		
	'''
	slices_d = [items[i::k] for i in range(k)]
	slices_l = [labels[i::k] for i in range(k)]

	for i in range(k):
		validation_label = slices_l[i]
		validation = slices_d[i]
		training = [item
		            for s in slices_d if s is not validation
		            for item in s]
		training_label = [labels
		            for s in slices_l if s is not validation_label
		            for labels in s]
		yield training, validation, training_label, validation_label



def accuracy2(w, train_data, labels):
	'''Measures the accuracy for each validation set.
		accuracy measure is = 100*number of correct class classification/ total number of test data  
	'''
	train_data = np.asarray(train_data)
	labels = np.asarray(labels)
	l_0 = train_data
	l_1 = sigmoid(np.dot(l_0,w[0]))
	l_2 = sigmoid(np.dot(l_1,w[1]))
	l_3 = sigmoid(np.dot(l_2,w[2]))
	l_4 = sigmoid(np.dot(l_3,w[3]))
	l_5 = sigmoid(np.dot(l_4,w[4]))
	#l_3 = l_3
	k=0
	for i in range(len(l_5)):

		if list(l_5[i]).index(max(l_5[i])) == list(labels[i]).index(max(labels[i])):
			k +=1
			#print(k)
	acc = k/len(l_5)*100
	print("accuracy = {0} %".format(acc))

	return l_5, acc


def predict(w, test_data, labels,file):
	'''Pedicts the weights of the test set using the trained weights. W used here is the best updated 
		weight from the validation reults.
		This saves the output classes in csv files.
	'''
	train_data = np.asarray(test_data)
	labels = np.asarray(labels)
	l_0 = train_data
	l_1 = sigmoid(np.dot(l_0,w[0]))
	l_2 = sigmoid(np.dot(l_1,w[1]))
	l_3 = sigmoid(np.dot(l_2,w[2]))
	l_4 = sigmoid(np.dot(l_3,w[3]))
	l_5 = sigmoid(np.dot(l_4,w[4]))

	predict = []
	for i in range(len(l_5)):
		predict.append(list(l_5[i]).index(max(l_5[i])))

	id_ = [i for i in range(len(l_5))]

	with open(file,'w') as f:
		writer = csv.writer(f, delimiter = ',')
		f.write("id,predicted_class")
		writer.writerows(list(zip(id_,predict)))