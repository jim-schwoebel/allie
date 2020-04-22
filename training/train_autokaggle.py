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

def train_autokaggle(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):
	print('initializing installation')
	os.system('pip3 install autokaggle==0.1.0')
		
	# training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)

	if mtype in ['classification', 'c']:

		modelname=jsonfile[0:-5]+'_autokaggle_classification_'+str(default_features).replace("'",'').replace('"','')
		model_name=modelname+'.pickle'
		jsonfilename=modelname+'.json'


		clf = TabularClassifier()
		clf.fit(X_train, y_train, time_limit=12 * 60 * 60)

		# stats
		F1_score = clf.evaluate(X_test, y_test)
		label_predictions = clf.predict(X_test)
		accuracy=accuracy_score(y_test, label_predictions)

		print('F1 score - ', F1_score)
		print('Accuracy - ', accuracy)

		# SAVE ML MODEL
		modelfile=open(model_name,'wb')
		pickle.dump(clf, modelfile)
		modelfile.close()

		# SAVE JSON FILE 
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'accuracy':accuracy,
			'F1 score': F1_score,
			'model type':'autokaggle_classification',
			'settings': settings,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

	elif mtype in ['regression', 'r']:

		modelname=jsonfile[0:-5]+'_gama_regression_'+str(default_features).replace("'",'').replace('"','')
		model_name=modelname+'.pickle'
		jsonfilename=modelname+'.json'

		print("Starting AutoKaggle")

		clf = TabularRegressor()
		clf.fit(X_train, y_train, time_limit=12 * 60 * 60)
		mse_error = clf.evaluate(X_test, y_test)
		print('MSE - ', mse_error)

		# saving model
		print('saving model')
		modelfile=open(model_name,'wb')
		pickle.dump(clf, modelfile)
		modelfile.close()

		# save JSON
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'mse_error':mse_error,
			'model type':'autokaggle_regression',
			'settings': settings,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

	# move model to proper directory
	cur_dir2=os.getcwd()
	try:
		os.chdir(problemtype+'_models')
	except:
		os.mkdir(problemtype+'_models')
		os.chdir(problemtype+'_models')

	# now move all the files over to proper model directory 
	shutil.move(cur_dir2+'/'+jsonfilename, os.getcwd()+'/'+jsonfilename)
	shutil.move(cur_dir2+'/'+model_name, os.getcwd()+'/'+model_name)

	model_dir=os.getcwd()

	return model_name, model_dir
