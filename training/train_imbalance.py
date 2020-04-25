import warnings, datetime, uuid, os, json, shutil, pickle

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pandas as pd


'''
Taken from the example here:
https://imbalanced-learn.readthedocs.io/en/stable/

Plotting taken from:
https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html
'''

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

def train_imbalance(alldata, labels, mtype, jsonfile, problemtype, default_features, settings, distributions):
	print('installing package configuration')
	os.system('pip3 install -U imbalanced-learn')
	# os.system('pip3 install scipy==1.4.1')
	# os.system('pip3 install scikit-learn==0.20.1')

	# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
	# License: MIT
	from collections import Counter
	from sklearn.datasets import load_iris
	from sklearn.svm import LinearSVC
	from sklearn.model_selection import train_test_split
	# from imblearn.datasets import make_imbalance
	from imblearn.under_sampling import NearMiss
	from imblearn.pipeline import make_pipeline
	from imblearn.metrics import classification_report_imbalanced

	# get train and test data
	print('creating training data')
	# we need to load the classes into this one to pull the dataset and load the imbalance
	
	# create file names
	model_name=jsonfile[0:-5]+'_'+str(default_features).replace("'",'').replace('"','')+'_imbalancedlearn'

	if mtype == 'c':
		model_name=model_name+'_classification'
		mtype='classification'
	elif mtype == 'r':
		model_name=model_name+'_regression'
		mtype='regression'

	folder=model_name
	jsonfilename=model_name+'.json'
	csvfilename=model_name+'.csv'
	trainfile=model_name+'_train.csv'
	testfile=model_name+'_test.csv'
	model_name=model_name+'.pickle'

	# this should be the model directory
	hostdir=os.getcwd()

	# make dataset
	RANDOM_STATE = 42

	X_train, X_test, y_train, y_test = train_test_split(alldata, labels, random_state=RANDOM_STATE)

	print('Training target statistics: {}'.format(Counter(y_train)))
	print('Testing target statistics: {}'.format(Counter(y_test)))


	if mtype=='classification':

		# Create a pipeline
		pipeline = make_pipeline(NearMiss(version=2),
		                         LinearSVC(random_state=RANDOM_STATE))

		pipeline.fit(X_train, y_train)

		# Classify and report the results
		label_predictions = pipeline.predict(X_test)
		report=classification_report_imbalanced(y_test, label_predictions)
		print(report)
		accuracy=accuracy_score(y_test, label_predictions)
		print(accuracy)

		# now save the model in .pickle
		f=open(model_name,'wb')
		pickle.dump(pipeline, f)
		f.close()

		# SAVE JSON FILE 
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'accuracy': float(accuracy),
			'report': report,
			'model type':'imbalancedlearn_%s'%(mtype),
			'settings': settings,
			'distributions': distributions,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

	elif mtype == 'regression':

		from sklearn.model_selection import StratifiedKFold
		from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
		from imblearn.pipeline import make_pipeline
		from sklearn.linear_model import LogisticRegression
		

		# Create a pipeline (usually works well with logistic regression 2 classes)
		pipeline = make_pipeline(SMOTE(random_state=RANDOM_STATE),
		                         LogisticRegression(random_state=0))
		pipeline.fit(X_train, y_train)
		predictions=pipeline.predict(X_test)
		mse_error=mean_squared_error(y_test,predictions)
		r2_score_=r2_score(y_test, predictions)

		print('MSE_Error')
		print(mse_error)
		print('R^2')
		print(r2_score_)

		# now save the model in .pickle
		f=open(model_name,'wb')
		pickle.dump(pipeline, f)
		f.close()
		
		# save the .JSON file
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':model_name,
			'mse_error': mse_error,
			'r2_score': float(r2_score_),
			'model type':'imbalancedlearn_%s'%(mtype),
			'settings': settings,
			'distributions': distributions,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

	# tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)

	# now get all them transferred
	os.chdir(hostdir)
	try:
		os.chdir(problemtype+'_models')
	except:
		os.mkdir(problemtype+'_models')
		os.chdir(problemtype+'_models')

	# now move all the files over to proper model directory 
	shutil.copy(hostdir+'/'+model_name, hostdir+'/%s_models/%s'%(problemtype,model_name))
	shutil.copy(hostdir+'/'+jsonfilename, hostdir+'/%s_models/%s'%(problemtype,jsonfilename))

	# get variables
	model_dir=hostdir+'/%s_models/'%(problemtype)

	return model_name, model_dir
