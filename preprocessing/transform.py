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

def get_features(classes, problem_type):

	features=list()
	feature_labels=list()
	class_labels=list()
	curdir=os.getcwd()

	for i in range(len(classes)):

		print('----------LOADING %s----------'%(classes[i].upper()))
		os.chdir(curdir+'/'+classes[i])
		listdir=os.listdir()
		jsonfiles=list()

		for j in range(len(listdir)):
			if listdir[j].endswith('.json'):
				jsonfiles.append(listdir[j])

		g=json.load(open(jsonfiles[0]))
		feature_list=list(g['features'][problem_type])

		for j in tqdm(range(len(jsonfiles))):
			g=json.load(open(jsonfiles[j]))
			feature_=list()
			label_=list()
			try:
				for k in range(len(feature_list)):
					feature_=feature_+g['features'][problem_type][feature_list[k]]['features']
					label_=label_+g['features'][problem_type][feature_list[k]]['labels']

				# quick quality check to only add to list if the feature_labels match in length the features_
				if len(feature_) == len(label_):
					features.append(feature_)
					feature_labels.append(label_)
					class_labels.append(classes[i])
			except:
				print('error loading feature embedding: %s'%(feature_list[k].upper()))


	return features, feature_labels, class_labels 


def feature_scale(feature_scaler, X_train, y_train):

	# more information about these scalers can be found @ 
	# https://scikit-learn.org/stable/modules/preprocessing.html

	if feature_scaler == 'binarizer':
		# scale the X values in the set 
		binarizer = preprocessing.Binarizer()
		binarizer.fit(X_train)

	elif feature_scaler == 'one_hot_encoder':
		'''
		>>> enc.transform([['female', 'from US', 'uses Safari'],
			             	['male', 'from Europe', 'uses Safari']]).toarray()
			array([[1., 0., 0., 1., 0., 1.],
			       [0., 1., 1., 0., 0., 1.]])
		'''
		# This is on y values 
		enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
		enc.fit(y_train)

	elif feature_scaler == 'minmaxscaler':
		X_scale = preprocessing.MinMaxScaler(X_train)

	elif feature_scaler == 'normalize':
		X_normalized = preprocessing.normalize(X_train, norm='l2')

	elif feature_scaler == 'power_transformer':
		# scale the X values in the set 
		pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
		X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))
		pt.fit_transform(X_lognormal)

	elif feature_scaler == 'poly':
		# scale the X values in the set 
		poly = PolynomialFeatures(2)
		poly.fit_transform(X_train)

	elif feature_scaler == 'quantile_transformer':
		# scale the X values in the set 
		quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
		quantile_transformer.fit(X_train)	
		
	elif feature_scaler == 'standard_scaler':
		# scale the X values in the set 
		X_scale = preprocessing.scale(X_train)

################################################
##	    		Load main settings    		  ##
################################################

# directory=sys.argv[1]
basedir=os.getcwd()
settingsdir=prev_dir(basedir)
print(settingsdir)
settings=json.load(open(settingsdir+'/settings.json'))
os.chdir(basedir)

# feature_scaler=settings['feature_scaler']
# default_audio_transformer=settings['default_feature_scaler']

# try:
	# assume 1 type of feature_set 
	# feature_scaler=[sys.argv[2]]
# except:
	# if none provided in command line, then load deafult features 
	# feature_scaler=settings['default_audio_transformers']

################################################
##	    	Now go featurize!                 ##
################################################

# get current directory 
curdir=os.getcwd()
basedir=prev_dir(curdir)
os.chdir(basedir+'/train_dir')
problem_type=sys.argv[1]

classes=get_classes()
features, feature_labels, class_labels = get_features(classes, problem_type)
x_train, x_test, y_train, y_test = train_test_split(features, class_labels, train_size=0.750, test_size=0.250)

for i in range(len(feature_scalers)):
	model=feature_scale(feature_scaler, x_train, y_train)
	
print(features[0])
print(feature_labels[1])
print(class_labels[0])
