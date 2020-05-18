import os, sys, shutil, pickle, json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

print('installing library')
os.system('pip3 install safe-transformer==0.0.5')
from SafeTransformer import SafeTransformer

def train_safe(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

	# only store transform and surrogate model
	model_name=common_name_model+'.pickle'
	files=list()
	curdir=os.getcwd()
	csvname=common_name_model.split('_')[0]
	
	# get training and testing data
	try:
		shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_train_transformed.csv',os.getcwd()+'/train.csv')
		shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_test_transformed.csv',os.getcwd()+'/test.csv')  
	except:
		shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_train.csv',os.getcwd()+'/train.csv') 
		shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_test.csv',os.getcwd()+'/test.csv')   

	# now load the training data as pandas dataframe
	data=pd.read_csv('train.csv')
	X_train=data.drop(columns=['class_'], axis=1)
	y_train=data['class_']

	print('Starting FIT')

	if mtype in ['classification', 'c']:
		print('CLASSIFICATION')
		print('training surrogate model...')
		surrogate_model = XGBClassifier().fit(X_train, y_train)
		print('training base model...')
		base_model = LogisticRegression().fit(X_train, y_train)
		safe_transformer = SafeTransformer(model=surrogate_model, penalty=1)
		pipe = Pipeline(steps=[('safe', safe_transformer), ('linear', base_model)])
		print('training pipeline...')
		pipe = pipe.fit(X_train, y_train)

	elif mtype in ['regression', 'r']:
		print('REGRESSION')
		surrogate_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,loss='huber')
		print('training surrogate model...')
		surrogate_model = surrogate_model.fit(X_train, y_train)
		print('loading base model')
		linear_model = LinearRegression()
		safe_transformer = SafeTransformer(surrogate_model, penalty = 0.84)
		print('training pipeline...')
		pipe = Pipeline(steps=[('safe', safe_transformer), ('linear', linear_model)])
		pipe = pipe.fit(X_train, y_train)

	# SAVE SURROGATE ML MODEL
	modelfile=open(model_name,'wb')
	pickle.dump(pipe, modelfile)
	modelfile.close()

	files.append(model_name)	
	files.append('train.csv')
	files.append('test.csv')
	model_dir=os.getcwd()

	return model_name, model_dir, files