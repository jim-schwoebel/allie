'''
Feature selectors:
- NNI - # https://nni.readthedocs.io/en/latest/FeatureEngineering/GradientFeatureSelector.html
- Scikit-Learn - 
'''

import json, os, sys
import numpy as np 
from sklearn.model_selection import train_test_split

def prev_dir(directory):
	g=directory.split('/')
	dir_=''
	for i in range(len(g)):
		if i != len(g)-1:
			if i==0:
				dir_=dir_+g[i]
			else:
				dir_=dir_+'/'+g[i]
	# print(dir_)
	return dir_

def feature_reduce(dimensionality_selector, X_train, y_train, component_num):

	if dimensionality_selector == 'autoencoder':

		from keras.layers import Input, Dense
		from keras.models import Model
		from sklearn.preprocessing import LabelEncoder
		from sklearn.preprocessing import OneHotEncoder

		# preprocess labels (make into integers)
		label_encoder = LabelEncoder()
		y_train=label_encoder.fit_transform(y_train)
		y_test=label_encoder.fit_transform(y_test)

		# this is the size of our encoded representations (208 features in X)
		encoding_dim = 32
		# add a few dimensions for encoder and decoder 
		input_dim = Input(shape=X_train[0].shape)
		encoder=Dense(encoding_dim, activation='tanh')
		autoencoder = Model(input_dim, decoded)
		# this model maps an input to its encoded representation
		encoder = Model(input_dim, encoded)
		# create a placeholder for an encoded (50-dimensional) input
		encoded_input = Input(shape=(encoding_dim,))
		# retrieve the last layer of the autoencoder model
		decoder_layer = autoencoder.layers[-1]
		# create the decoder model
		decoder = Model(encoded_input, decoder_layer(encoded_input))
		# now train autoencoder 
		autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
		autoencoder.fit(X_train, y_train,
		                epochs=50,
		                batch_size=256,
		                shuffle=True,
		                validation_data=(X_test, y_test))
		# predict emebddings
		encoded_audio = encoder.predict(X_test)
		decoded_audio = decoder.predict(encoded_audio)

		print('not saving model due to keras autoencoder')

	elif dimensionality_selector == 'cca':
		from sklearn.cross_decomposition import CCA
		cca = CCA(n_components=component_num)
		return cca

	elif dimensionality_selector == 'dictionary':
		from sklearn.decomposition import MiniBatchDictionaryLearning
		dico_X = MiniBatchDictionaryLearning(n_components=component_num, alpha=1, n_iter=500)
		model=dico_X

	elif dimensionality_selector == 'ica':
		from sklearn.decomposition import FastICA
		ica = FastICA(n_components=component_num)
		model=ica

	elif dimensionality_selector == 'kmeans':
		from sklearn.cluster import KMeans
		kmeans = KMeans(n_clusters=component_num, random_state=0)
		model=kmeans

	elif dimensionality_selector == 'lda':
		from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
		lda = LDA(n_components=component_num).fit(X_train, y_train).transform(X_train)
		model=lda

	elif dimensionality_selector == 'manifold':
		from sklearn import manifold
		manifold_X = manifold.Isomap(10, component_num)
		model=manifold_X

	elif dimensionality_selector == 'neighborhood':
		from sklearn.neighbors import NeighborhoodComponentsAnalysis
		nca = NeighborhoodComponentsAnalysis(random_state=42)
		model=nca

	# feature_engineering.gradient.selector 
	elif dimensionality_selector == 'pca':
		from sklearn.decomposition import PCA
		pca = PCA(n_components=component_num)
		model = pca

	elif dimensionality_selector == 'pls':
		from sklearn.cross_decomposition import PLSRegression
		pls = PLSRegression(n_components=component_num)
		model=pls

	return model
