import os, sys, pickle, json, random, shutil, time, yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import time

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

	data['class']=y_train
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

def edit_modelfile(data_,problemtype,csvfilename):
	# open the yml file
	list_doc=yaml.load(open("model.yml"), Loader=yaml.FullLoader)
	os.remove('model.yml')

	# load sections / format and modify appropriately
	# ----> just change file name
	project=list_doc['project']
	project['submission_file']=csvfilename[0:-4]

	# ----> set right target value here
	# { 'features': '*', 'sampling': {'option': False, 'method': 'under_random', 'ratio': 0.0}, 'sentinel': -1, 'separator': ',', 'shuffle': False, 'split': 0.4, 'target': 'won_on_spread', 'target_value': True}
	data=list_doc['data']
	print(data)
	data['drop']=['Unnamed: 0']
	data['shuffle']=True
	data['split']=0.4
	data['target']='class'

	# ----> now set right model parameters here
	model=list_doc['model']
	# {'algorithms': ['RF', 'XGB'], 'balance_classes': False, 'calibration': {'option': False, 'type': 'isotonic'}, 'cv_folds': 3, 'estimators': 201, 'feature_selection': {'option': False, 'percentage': 50, 'uni_grid': [5, 10, 15, 20, 25], 'score_func': 'f_classif'}, 'grid_search': {'option': True, 'iterations': 50, 'random': True, 'subsample': False, 'sampling_pct': 0.25}, 'pvalue_level': 0.01, 'rfe': {'option': True, 'step': 5}, 'scoring_function': 'roc_auc', 'type': 'classification'}
	if problemtype in ['classification', 'c']:
		model['algorithms']=['AB','GB','KERASC','KNN','LOGR','LSVC','LSVM','NB','RBF','RF','SVM','XGB','XGBM','XT']
		model['scoring_function']='roc_auc'
		model['type']='classification'

	elif problemtype in ['regression','r']:
		model['algorithms']=['GBR','KERASR','KNR','LR','RFR','XGBR','XTR']
		model['scoring_function']='mse'
		model['type']='regression'

	# just remove the target class
	features=list_doc['features']
	features['factors']=list(data_).remove('class')

	# everything else remains the same
	pipeline=list_doc['pipeline']
	plots=list_doc['plots']
	xgboost_=list_doc['xgboost']

	# now reconfigure the doc
	list_doc['project']=project
	list_doc['data']=data
	list_doc['model']=model
	list_doc['features']=features
	list_doc['pipeline']=pipeline
	list_doc['plots']=plots
	# list_doc['xgboost']=xgboost

	print(list_doc)

	# now re-write the file
	print('re-writing YAML config file...')
	file=open("model.yml", 'w')
	yaml.dump(list_doc, file)
	file.close()

	print(list_doc)
	file.close()

def train_alphapy(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):
	# install required dependencies
	print('installing dependencies')
	os.system('pip3 install alphpy==2.4.0')
	os.system('pip3 install imbalance-learn==0.5.0')
	os.system('pip3 install xgboost==0.80')
	os.system('pip3 install pandas==1.0')
	os.system('pip3 install pandas-datareader==0.8')
	# os.system('pip3 install scikit-learn==0.20.1')

	# get train and test data
	print('creating training data')
	X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)

	# create file names
	jsonfilename=jsonfile[0:-5]+'_'+str(default_features).replace("'",'').replace('"','')+'_alphapy.json'
	csvfilename=jsonfilename[0:-5]+'.csv'
	picklefilename=jsonfilename[0:-5]+'.pickle'
	folder=jsonfilename[0:-5]

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

	try:
		os.mkdir(folder)
	except:
		shutil.rmtree(folder)
		os.mkdir(folder)

	basedir=os.getcwd()

	# make required directories
	os.chdir(folder)
	os.mkdir('data')
	os.mkdir('input')
	
	# now make a .CSV of all the data
	os.chdir('data')
	all_data = convert_(alldata, labels, labels_)
	train_data= convert_(X_train, y_train, labels_)
	test_data= convert_(X_test, y_test, labels_)
	all_data.to_csv(csvfilename)
	data=pd.read_csv(csvfilename)
	os.remove(csvfilename)

	os.chdir(basedir)
	os.chdir(folder)
	os.chdir('input')
	train_data.to_csv('train.csv')
	test_data.to_csv('test.csv')

	os.chdir(basedir)
	shutil.copytree(prev_dir(hostdir)+'/training/helpers/alphapy/config/', basedir+'/'+folder+'/config')
	os.chdir(folder)
	os.chdir('config')
	edit_modelfile(data, problemtype, csvfilename)

	os.chdir(basedir)
	os.chdir(folder)
	os.system('alphapy')
	os.chdir(hostdir)

	return model_name, model_dir
