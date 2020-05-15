import os, sys, shutil, pickle, json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
curdir=os.getcwd()
print(os.getcwd())
from helpers.autokaggle.tabular_supervised import TabularClassifier
from helpers.autokaggle.tabular_supervised import TabularRegressor
import numpy as np
os.chdir(curdir)

def train_autokaggle(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):
	print('initializing installation')
	os.system('pip3 install autokaggle==0.1.0')
	
	model_name=common_name_model+'.pickle'
	files=list()

	if mtype in ['classification', 'c']:

		# fit classifier 
		clf = TabularClassifier()
		clf.fit(X_train, y_train, time_limit=12 * 60 * 60)

		# SAVE ML MODEL
		modelfile=open(model_name,'wb')
		pickle.dump(clf, modelfile)
		modelfile.close()

	elif mtype in ['regression', 'r']:

		print("Starting AutoKaggle")
		clf = TabularRegressor()
		clf.fit(X_train, y_train, time_limit=12 * 60 * 60)

		# saving model
		print('saving model')
		modelfile=open(model_name,'wb')
		pickle.dump(clf, modelfile)
		modelfile.close()

	model_dir=os.getcwd()
	files.append(model_name)

	return model_name, model_dir, files