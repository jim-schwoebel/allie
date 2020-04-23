import os, json, shutil, pickle, sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_log_error
import torch
import sklearn.model_selection
import sklearn.metrics
import pandas as pd

print('installing library')
os.system('pip3 install autopytorch==0.0.2')

from autoPyTorch import AutoNetClassification

'''
From the documentation: https://github.com/automl/Auto-PyTorch
'''
# install mlblocks

def train_autopytorch(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):

	# name model
	modelname=jsonfile[0:-5]+'_autopytorch_'+str(default_features).replace("'",'').replace('"','')

	# training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)

	if mtype=='c': 

		# rename files with classification
		modelname=modelname+'_classification'
		model_name=modelname+'.pth'
		jsonfilename=modelname+'.json'

		# running Auto-PyTorch
		autonet = AutoNetClassification("tiny_cs",  # config preset
		                                    log_level='info',
		                                    max_runtime=600,
		                                    min_budget=30,
		                                    max_budget=90)

		autonet.fit(X_train, y_train, validation_split=0.3)
		y_pred = autonet.predict(X_test)

		accuracy=sklearn.metrics.accuracy_score(y_test, y_pred)
		print("Accuracy score", accuracy)
		pytorch_model = autonet.get_pytorch_model()
		print(pytorch_model)
		# saving model
		print('saving model')
		torch.save(pytorch_model, model_name)

		# SAVE JSON FILE 
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'accuracy':accuracy,
			'model type':'autopytorch_classification',
			'settings': settings,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

	if mtype=='r': 

		# rename files with regression
		modelname=modelname+'_regression'
		model_name=modelname+'.pickle'
		jsonfilename=modelname+'.json'

		# run model session
		sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
		from autoPyTorch import AutoNetRegression
		from autoPyTorch.data_management.data_manager import DataManager

		# Note: You can write your own datamanager! Call fit train, valid data (numpy matrices) 
		dm = DataManager()
		dm.generate_regression(num_features=21, num_samples=1500)

		# Note: every parameter has a default value, you do not have to specify anything. The given parameter allow a fast test.
		autonet = AutoNetRegression(budget_type='epochs', min_budget=1, max_budget=9, num_iterations=1, log_level='info')
		pipeline = autonet.fit(X_train, y_train)
		y_pred = pipeline.predict(X_test)
		mse_error=mean_squared_log_error(y_test, y_pred)

		print('MSE: ', mse_error)

		# saving model
		print('saving model')
		modelfile=open(model_name,'wb')
		pickle.dump(pipeline, modelfile)
		modelfile.close()

		# save JSON
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'mse_error':mse_error,
			'model type':'autopytorch_regression',
			'settings': settings,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

	cur_dir2=os.getcwd()
	try:
		os.chdir(problemtype+'_models')
	except:
		os.mkdir(problemtype+'_models')
		os.chdir(problemtype+'_models')

	# now move all the files over to proper model directory 
	shutil.copy(cur_dir2+'/'+model_name, os.getcwd()+'/'+model_name)
	shutil.copy(cur_dir2+'/'+jsonfilename, os.getcwd()+'/'+jsonfilename)
	os.remove(cur_dir2+'/'+model_name)
	os.remove(cur_dir2+'/'+jsonfilename)

	# get model directory
	model_dir=os.getcwd()

	return model_name, model_dir