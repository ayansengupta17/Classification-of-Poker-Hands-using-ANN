import numpy as np


def get_labels(train):
	'''Extract the labels from the input data and return the labels along 
	 train data table without labels.'''
	output = np.array(train)[:, -1].reshape(1, len(train))
	train = train.drop('class', axis=1)

	return output, train


def encode_class(labels, n):
	''' Encode the labels ie classes to form lists 
		eg:
		class 0 after encoding : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		class 1 after encoding : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
		class 2 after encoding : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
		class 3 after encoding : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
		class 4 after encoding : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
		class 5 after encoding : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
		class 6 after encoding : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
		class 7 after encoding : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
		class 8 after encoding : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
		class 9 after encoding : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
	'''
	onehot_encoded = list()
	for value in labels:
		l = [0 for _ in range(n)]
		l[value] = 1
		onehot_encoded.append(l)
	return onehot_encoded


def encode_train(train):
	''' Encode the suite and card values 
	Different suites would be encoded like
	[1, 0, 0, 0]
	[0, 1, 0, 0]
	[0, 0, 1, 0]
	[0, 0, 0, 1]

	Different card values would be encoded like:
	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
	[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
	[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
	[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

	So a typical hand  [s1,c1,s2,c2,s3,c3,s4,c4,s5,c5] would be a list of length 85

	example:

	A typical hand like	[4,7,3,5,3,3,1,13,4,8,0] would be encoded to 

	array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
	0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,
	0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
	0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
	0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
	0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
	0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.])

	'''

	onehot_encoded = list()

	for k in range(len(train)):
		s1 = [0]*4
		c1 = [0]*13
		s2 = [0]*4
		c2 = [0]*13
		s3 = [0]*4
		c3 = [0]*13
		s4 = [0]*4
		c4 = [0]*13
		s5 = [0]*4
		c5 = [0]*13
		hand = [s1,c1,s2,c2,s3,c3,s4,c4,s5,c5]
		_ = hand[:]

		for i,v in enumerate(train[k]):
			_[i][v-1] = 1

		hand_ = sum(_, [])
		onehot_encoded.append(hand_)



	return onehot_encoded