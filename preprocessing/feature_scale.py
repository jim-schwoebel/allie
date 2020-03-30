'''
Import all the featurization scripts and allow the user to customize what embedding that
they would like to use for modeling purposes.

AudioSet is the only embedding that is a little bit wierd, as it is normalized to the length
of each audio file. There are many ways around this issue (such as normalizing to the length 
of each second), however, I included all the original embeddings here in case the time series
information is useful to you.
'''

import json, os, sys
import numpy as np 
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

def get_classes():
	count=2
	classes=list()
	while True:
		try:
			class_=sys.argv[count]
			classes.append(class_)
			print(classes)
			count=count+1
		except:
			break

	return classes

def feature_scale(feature_scaler, X_train, y_train):

	# more information about these scalers can be found @ 
	# https://scikit-learn.org/stable/modules/preprocessing.html

	if feature_scaler == 'binarizer':
		# scale the X values in the set 
		model = preprocessing.Binarizer()

	elif feature_scaler == 'one_hot_encoder':
		'''
		>>> enc.transform([['female', 'from US', 'uses Safari'],
			             	['male', 'from Europe', 'uses Safari']]).toarray()
			array([[1., 0., 0., 1., 0., 1.],
			       [0., 1., 1., 0., 0., 1.]])
		'''
		# This is on y values 
		model = preprocessing.OneHotEncoder(handle_unknown='ignore')

	elif feature_scaler == 'maxabs':
		model=preprocessing.MaxAbsScaler()

	elif feature_scaler == 'minmax':
		model=preprocessing.MinMaxScaler()

	elif feature_scaler == 'normalize':
		# L2 normalization
		model = preprocessing.Normalizer()

	elif feature_scaler == 'poly':
		# scale the X values in the set 
		model = PolynomialFeatures(2)

	elif feature_scaler == 'power_transformer':
		# scale the X values in the set 
		model = preprocessing.PowerTransformer(method='yeo-johnson')

	elif feature_scaler == 'quantile_transformer_normal':
		# scale the X values in the set 
		model = preprocessing.QuantileTransformer(output_distribution='normal')

	elif feature_scaler == 'robust':
		model=preprocessing.RobustScaler(quantile_range=(25, 75)) 

	elif feature_scaler == 'standard_scaler':
		# scale the X values in the set 
		model=preprocessing.StandardScaler()

	return model 