import os, sys, pickle, json, random, shutil, time, yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
import pandas as pd
import time

'''

Saving and loading autokeras models 

for autokeras 1.0:
save the model
model.save('my_model')

load the model
import tensorflow as tf
new_model = tf.keras.models.load_model('my_model')

'''
def convert_(X_train, y_train, labels):

	feature_list=labels
	data=dict()

	print(len(feature_list))
	print(len(X_train[0]))
	# time.sleep(50)

	for i in range(len(X_train)):
		for j in range(len(feature_list)-1):
			if i > 0:
				# print(data[feature_list[j]])
				try:
					# print(feature_list[j])
					# print(data)
					# print(X_train[i][j])
					# print(data[feature_list[j]])
					# time.sleep(2)
					data[feature_list[j]]=data[feature_list[j]]+[X_train[i][j]]
				except:
					pass
					# print(data)
					# time.sleep(50)
					# print(str(i)+'-i')
					# print(j)

			else:
				data[feature_list[j]]=[X_train[i][j]]
				print(data)

	data['class_']=y_train
	data=pd.DataFrame(data, columns = list(data))
	print(data)
	print(list(data))
	# time.sleep(500)

	return data

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

def train_autokeras(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):
	print('installing package')
	os.system('pip3 install tensorflow==2.1.0')
	os.system('pip3 install autokeras==1.0')

	import autokeras as ak
	import tensorflow as tf

	# get train and test data
	print('creating training data')
	X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)

	# create file names
	model_name=jsonfile[0:-5]+'_'+str(default_features).replace("'",'').replace('"','')+'_autokeras'

	if mtype == 'c':
		model_name=model_name+'_classification'
		mtype='classification'
	elif mtype == 'r':
		model_name=model_name+'_regression'
		mtype='regression'

	jsonfilename=model_name+'.json'
	csvfilename=model_name+'.csv'
	folder=model_name
	trainfile=model_name+'_train.csv'
	testfile=model_name+'_test.csv'
	model_name=model_name+'.h5'

	# this should be the model directory
	hostdir=os.getcwd()

	# open a sample featurization 
	labels_dir=prev_dir(hostdir)+'/train_dir/'+jsonfilename.split('_')[0]
	os.chdir(labels_dir)
	listdir=os.listdir()
	features_file=''
	for i in range(len(listdir)):
		if listdir[i].endswith('.json'):
			features_file=listdir[i]

	# load features file and get labels 
	labels_=json.load(open(features_file))['features'][problemtype][default_features]['labels']
	os.chdir(hostdir)

	# make a temporary folder for the training session
	try:
		os.mkdir(folder)
		os.chdir(folder)
	except:
		shutil.rmtree(folder)
		os.mkdir(folder)
		os.chdir(folder)

	all_data = convert_(alldata, labels, labels_)
	train_data= convert_(X_train, y_train, labels_)
	test_data= convert_(X_test, y_test, labels_)
	all_data.to_csv(csvfilename)
	data=pd.read_csv(csvfilename)
	os.remove(csvfilename)
	train_data.to_csv(trainfile)
	test_data.to_csv(testfile)

	if mtype=='classification':
		# Initialize the classifier.
		# 30 trials later
		clf = ak.StructuredDataClassifier(max_trials=3)
		# x is the path to the csv file. y is the column name of the column to predict.
		clf.fit(X_train, y_train)
		# Evaluate the accuracy of the found model.
		predictions=clf.predict(X_test)	
		accuracy=accuracy_score(y_test, predictions)
		print('------------------------------------')
		print('	BEST MODEL 	 			')
		print('------------------------------------')
		print('ACCURACY')
		print(accuracy)
		time.sleep(3)
		# saving model
		# debugged from stackoverflow: https://stackoverflow.com/questions/59533584/how-to-save-load-models-in-autokeras-1-0
		print('saving model architecture')
		best_model = clf.tuner.get_best_model()[1]
		print(best_model)
		print(type(best_model))
		print(model_name)
		time.sleep(5)
		best_model.save(model_name)
		json_string = best_model.to_json()

		# SAVE JSON FILE 
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'accuracy': float(accuracy),
			'model type':'autokeras_%s'%(mtype),
			'settings': settings,
			'architecture': json_string,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

		model = clf.export_model()

	elif mtype == 'regression':

		# change to 100 trials 
		train_dataset=pd.read_csv(trainfile)
		regressor = ak.StructuredDataRegressor(max_trials=3)
		regressor.fit(X_train, y_train)
		# Evaluate the accuracy of the found model.
		predictions=regressor.predict(X_test)		
		mse_error=mean_squared_error(y_test, predictions)
		print('------------------------------------')
		print('	BEST MODEL 	 			')
		print('------------------------------------')
		print('MSE-ERROR')
		print(mse_error)
		time.sleep(3)
		# saving model
		# debugged from stackoverflow: https://stackoverflow.com/questions/59533584/how-to-save-load-models-in-autokeras-1-0
		print('saving model architecture')
		best_model = regressor.tuner.get_best_model()[1]
		print(best_model)
		print(type(best_model))
		print(model_name)
		time.sleep(5)
		best_model.save(model_name)
		json_string = best_model.to_json()

		# SAVE JSON FILE 
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'mse_error': float(mse_error),
			'model type':'autokeras_%s'%(mtype),
			'settings': settings,
			'architecture': json_string,
		}

		json.dump(data,jsonfile)
		jsonfile.close()
		model = regressor.export_model()

	# tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)

	# now get all them transferred
	os.chdir(hostdir)
	try:
		os.chdir(problemtype+'_models')
	except:
		os.mkdir(problemtype+'_models')
		os.chdir(problemtype+'_models')

	# now move all the files over to proper model directory 
	shutil.copy(hostdir+'/'+folder+'/'+model_name, hostdir+'/%s_models/%s'%(problemtype,model_name))
	shutil.copy(hostdir+'/'+folder+'/'+jsonfilename, hostdir+'/%s_models/%s'%(problemtype,jsonfilename))
	shutil.copytree(hostdir+'/'+folder, hostdir+'/%s_models/'%(problemtype)+folder)
	os.remove(hostdir+'/'+folder+'/'+model_name)
	os.remove(hostdir+'/'+folder+'/'+jsonfilename)
	shutil.rmtree(hostdir+'/'+folder)

	# get variables
	model_dir=hostdir+'/%s_models/'%(problemtype)

	return model_name, model_dir
