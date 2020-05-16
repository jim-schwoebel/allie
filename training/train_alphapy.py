import os

# install required dependencies (important to have this config for everything to work)
print('installing dependencies')
# 2.4.0 before
os.system('pip3 install alphapy==2.4.2')
os.system('pip3 install imbalanced-learn==0.5.0')
os.system('pip3 install pandas==1.0')
os.system('pip3 install pandas-datareader==0.8.1')
os.system('pip3 install xgboost==0.80')

import sys, pickle, json, random, shutil, time, yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def convert_(X_train, y_train, labels):
	# create proper training data
	feature_list=labels
	data=dict()

	for i in range(len(X_train)):
		for j in range(len(feature_list)-1):
			if i > 0:
				try:
					data[feature_list[j]]=data[feature_list[j]]+[X_train[i][j]]
				except:
					pass

			else:
				data[feature_list[j]]=[X_train[i][j]]
				print(data)

	data['class_']=y_train
	data=pd.DataFrame(data, columns = list(data))
	print(data)
	print(list(data))

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

def edit_modelfile(data_,mtype,csvfilename):
	# open the yml file
	list_doc=yaml.load(open("model.yml"), Loader=yaml.Loader)
	os.remove('model.yml')

	# load sections / format and modify appropriately
	# ----> just change file name
	project=list_doc['project']
	# project['submission_file']=csvfilename[0:-4]

	# ----> set right target value here
	# { 'features': '*', 'sampling': {'option': False, 'method': 'under_random', 'ratio': 0.0}, 'sentinel': -1, 'separator': ',', 'shuffle': False, 'split': 0.4, 'target': 'won_on_spread', 'target_value': True}
	data=list_doc['data']
	print(data)
	data['drop']=['Unnamed: 0']
	data['shuffle']=True
	data['split']=0.4
	data['target']='class_'

	# ----> now set right model parameters here
	model=list_doc['model']
	# {'algorithms': ['RF', 'XGB'], 'balance_classes': False, 'calibration': {'option': False, 'type': 'isotonic'}, 'cv_folds': 3, 'estimators': 201, 'feature_selection': {'option': False, 'percentage': 50, 'uni_grid': [5, 10, 15, 20, 25], 'score_func': 'f_classif'}, 'grid_search': {'option': True, 'iterations': 50, 'random': True, 'subsample': False, 'sampling_pct': 0.25}, 'pvalue_level': 0.01, 'rfe': {'option': True, 'step': 5}, 'scoring_function': 'roc_auc', 'type': 'classification'}
	if mtype in ['classification', 'c']:
		model['algorithms']=['AB','GB','KNN','LOGR','RF','XGB','XT'] # removed 'KERASC', 'LSVC', 'LSVM', 'NB', 'RBF', 'SVM', 'XGBM'
		model['scoring_function']='roc_auc'
		model['type']='classification'

	elif mtype in ['regression','r']:
		model['algorithms']=['GBR','KNR','LR','RFR','XGBR','XTR'] # remove 'KERASR'
		model['scoring_function']='mse'
		model['type']='regression'

	# just remove the target class
	features=list_doc['features']
	# features['factors']=list(data_).remove('class_')

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

def train_alphapy(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

	# create file names
	csvfilename=common_name_model+'.csv'
	picklefilename=common_name_model+'.pickle'
	folder=common_name_model+'_session'
	csvname=common_name_model.split('_')[0]

	# files
	files=list()

	# this should be the model directory
	hostdir=os.getcwd()

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
	try:
		shutil.copy(hostdir+'/'+model_session+'/data/'+csvname+'_all_transformed.csv',os.getcwd()+'/'+csvfilename)
	except:
		shutil.copy(hostdir+'/'+model_session+'/data/'+csvname+'_all.csv',os.getcwd()+'/'+csvfilename)
	data=pd.read_csv(csvfilename)
	os.remove(csvfilename)

	os.chdir(basedir)
	os.chdir(folder)
	os.chdir('input')
	try:
		shutil.copy(hostdir+'/'+model_session+'/data/'+csvname+'_train_transformed.csv',os.getcwd()+'/train.csv')
		shutil.copy(hostdir+'/'+model_session+'/data/'+csvname+'_test_transformed.csv',os.getcwd()+'/test.csv')
	except:
		shutil.copy(hostdir+'/'+model_session+'/data/'+csvname+'_train.csv',os.getcwd()+'/train.csv')
		shutil.copy(hostdir+'/'+model_session+'/data/'+csvname+'_test.csv',os.getcwd()+'/test.csv')		
	
	os.chdir(basedir)
	shutil.copytree(prev_dir(hostdir)+'/training/helpers/alphapy/config/', basedir+'/'+folder+'/config')
	os.chdir(folder)
	os.chdir('config')
	edit_modelfile(data, mtype, csvfilename)

	os.chdir(basedir)
	os.chdir(folder)
	os.system('alphapy')
	os.chdir(hostdir)

	# get variables
	model_name=folder
	model_dir=os.getcwd()
	files.append(folder)

	return model_name, model_dir, files
