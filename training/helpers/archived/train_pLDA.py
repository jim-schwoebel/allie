'''
PLDA implementation from
https://github.com/RaviSoji/plda/blob/master/mnist_demo/mnist_demo.ipynb
'''

import os, sys, pickle 
import helpers.plda.plda as plda 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def train_pLDA(alldata, labels):
	# get train and test data 
	training_data, testing_data, training_labels, testing_labels = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)

	training_data = training_data.reshape(training_data.shape)
	testing_data = testing_data.reshape(testing_data.shape)

	# optimize number of principal components in terms of accuracy
	acclist=list()
	compnum=list()
	for i in range(2, len(training_data[0]),1):
		#try:
		classifier = plda.Classifier()
		numcomponents=i
		classifier.fit_model(training_data, training_labels, n_principal_components=numcomponents)
		predictions, log_p_predictions = classifier.predict(testing_data)
		accuracy=(testing_labels == predictions).mean()
		print(accuracy)
		if accuracy > 1:
			pass
		else:
			acclist.append(accuracy)
			print(accuracy)
			print(i)
		#except:
			# if dimension too high, break it 
			#print('error')

	maxacc=max(acclist)
	numcomponents=acclist.index(maxacc)+1 

	# now retrain with proper parameters 
	classifier = plda.Classifier()
	classifier.fit_model(training_data, training_labels, n_principal_components=numcomponents)
	predictions, log_p_predictions = classifier.predict(testing_data)
	accuracy=(testing_labels == predictions).mean()
	print('max acc %s with %s components'%(maxacc, numcomponents))

	Psi = classifier.model.Psi
	A = classifier.model.A
	inv_A = classifier.model.inv_A
	m = classifier.model.m

	# Indices of the subspace used for classification.
	relevant_U_dims = classifier.model.relevant_U_dims

	# # Prior Gaussian Parameters
	# classifier.model.prior_params.keys()
	# # Posterior Gaussian Parameters
	# classifier.model.posterior_params.keys()
	# classifier.model.posterior_params[0].keys()
	# # Posterior Predictive Gaussian Parameters
	# classifier.model.posterior_predictive_params.keys()
	# classifier.model.posterior_predictive_params[0].keys()

	'''
	Transforming Data to PLDA Space
	There are 4 "spaces" that result from the transformations the model performs:

	Data space ('D'),
	Preprocessed data space ('X'),
	Latent space ('U'), and
	The "effective" subspace of the latent space ('U_model'), which is essentially the set of dimensions the model actually uses for prediction.
	'''

	# U_model = classifier.model.transform(training_data, from_space='D', to_space='U_model')
	# print(training_data.shape)
	# print(U_model.shape)

	# D = classifier.model.transform(U_model, from_space='U_model', to_space='D')
	# print(U_model.shape)
	# print(D.shape)

	# create dump of classifier 
	print('saving classifier to disk')
	f=open('plda_classifier.pickle','wb')
	pickle.dump(classifier,f)
	f.close()