import os, sys, shutil, pickle, json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
from gama import GamaClassifier, GamaRegressor

def train_gama(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):
	# install library
	print('installing library')
	os.system('pip3 install gama==20.1.0')

	# training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)

	if mtype in ['classification', 'c']:

		modelname=jsonfile[0:-5]+'_gama_classification_'+str(default_features).replace("'",'').replace('"','')
		model_name=modelname+'.pickle'
		jsonfilename=modelname+'.json'

		automl = GamaClassifier(max_total_time=180, keep_analysis_log=None)
		print("Starting GAMA `fit` - usually takes around 3 minutes but can take longer for large datasets")
		automl.fit(X_train, y_train)

		label_predictions = automl.predict(X_test)
		probability_predictions = automl.predict_proba(X_test)

		accuracy=accuracy_score(y_test, label_predictions)
		log_loss_pred=log_loss(y_test, probability_predictions)
		log_loss_score=automl.score(X_test, y_test)

		print('accuracy:', accuracy)
		print('log loss pred:', log_loss_pred)
		print('log_loss_score', log_loss_score)

		# SAVE ML MODEL
		modelfile=open(model_name,'wb')
		pickle.dump(automl, modelfile)
		modelfile.close()

		# SAVE JSON FILE 
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'accuracy':accuracy,
			'log_loss_pred': log_loss_pred,
			'log_loss_score': log_loss_score,
			'model type':'gama_classification',
			'settings': settings,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

	elif mtype in ['regression', 'r']:

		modelname=jsonfile[0:-5]+'_gama_regression_'+str(default_features).replace("'",'').replace('"','')
		model_name=modelname+'.pickle'
		jsonfilename=modelname+'.json'

		automl = GamaRegressor(max_total_time=180, keep_analysis_log=None, n_jobs=1)
		print("Starting GAMA `fit` - usually takes around 3 minutes but can take longer for large datasets")
		automl.fit(X_train, y_train)

		predictions = automl.predict(X_test)
		mse_error=mean_squared_error(y_test, predictions)
		print("MSE:", mse_error)

		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'mse_error':mse_error,
			'model type':'gama_regression',
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
