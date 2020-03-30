'''
Make transformations in this order:

Feature scalers --> reduce dimensions --> select features.

A --> A`--> A`` --> A```

Can do this through scikit-leran pipelines.
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

################################################
##	    		Load main settings    		  ##
################################################

# directory=sys.argv[1]
basedir=os.getcwd()
settingsdir=prev_dir(basedir)
print(settingsdir)
settings=json.load(open(settingsdir+'/settings.json'))

# get all the important settings for the transformations 
scale_features=settings['scale_features']
reduce_dimensions=settings['reduce_dimensions']
select_featuers=settings['select_features']
default_scaler=settings['default_scaler']
default_reducer=settings['default_dimensionality_reducer']
default_selector=settings['default_feature_selector']

os.chdir(basedir)

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
x_train, x_test, y_train, y_test = train_test_split(features, class_labels, train_size=0.90, test_size=0.10)

for i in range(len(feature_scalers)):
	model=feature_scale(feature_scaler, x_train, y_train)
	
print(features[0])
print(feature_labels[1])
print(class_labels[0])

################################################
##	    	    Scale features               ##
################################################
if scale_features == True:
	import feature_scale as fsc_
	scaler_model=fsc_.feature_scale(feature_scaler, X_train, y_train)
################################################
##	    	   Reduce dimensions              ##
################################################
if reduce_dimensions == True:
	import feature_reduce as fre_
	dimension_model=fre_.feature_reduce(dimensionality_selector, X_train, y_train)
################################################
##	    	   Feature selection              ##
################################################
if select_features == True:
	import feature_select as fse_
	selection_model=fse_.feature_select(feature_selector, X_train, y_train)
