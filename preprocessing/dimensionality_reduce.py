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

def feature_select(dimensionality_selector, X_train, y_train):

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

	elif dimensionality_selector == 'cca':
		from sklearn.cross_decomposition import CCA
		cca = CCA(n_components=50).fit(X, Y).transform(X, Y)
		new_X=cca[0]
		new_Y=cca[1]

	elif dimensionality_selector == 'dictionary':
		from sklearn.decomposition import MiniBatchDictionaryLearning
		dico_X = MiniBatchDictionaryLearning(n_components=50, alpha=1, n_iter=500).fit_transform(X)
		dico_Y = MiniBatchDictionaryLearning(n_components=50, alpha=1, n_iter=500).fit_transform(Y)


	elif dimensionality_selector == 'ica':
		from sklearn.decomposition import FastICA
		ica = FastICA(n_components=50)
		S_ = ica.fit_transform(X_train)  # Reconstruct signals
		# The mixing matrix is analagous to PCA singular values
		A_ = ica.mixing_  # Get estimated mixing matrix


	elif dimensionality_selector == 'kmeans':
		from sklearn.cluster import KMeans
		kmeans = KMeans(n_clusters=50, random_state=0).fit_transform(X_train)


	elif feature_selector == 'lasso':
		from sklearn.cross_decomposition import PLSRegression
		pls = PLSRegression(n_components=50).fit(X, Y).transform(X, Y)
		pls_X=pls[0]
		pls_Y=pls[1]

	elif dimensionality_selector == 'lda':
		from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
		lda = LDA(n_components=50).fit(X_train, y_train).transform(X)

	elif dimensionality_selector == 'manifold':
		from sklearn import manifold
		manifold_X = manifold.Isomap(10, 50).fit_transform(X)
		manifold_Y = manifold.Isomap(10,50).fit_transform(Y)

	elif dimensionalit_selector == 'neighborhood':

		from sklearn.neighbors import NeighborhoodComponentsAnalysis
		nca = NeighborhoodComponentsAnalysis(random_state=42)
		nca.fit(X_train, y_train)
		nca.transform(X_train)
		
	# feature_engineering.gradient.selector 
	elif dimensionality_selector == 'pca':
		from sklearn.decomposition import PCA
		# calculate PCA for 50 components 
		pca = PCA(n_components=50)
		pca.fit(X_train)
		X_pca = pca.transform(X_train)
		print("PCA original shape:   ", X.shape)
		print("PCA transformed shape:", X_pca.shape)
		print(pca.explained_variance_ratio_)  
		print(np.sum(pca.explained_variance_ratio_))


