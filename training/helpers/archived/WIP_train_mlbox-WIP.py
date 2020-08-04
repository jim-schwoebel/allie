import os, json, shutil, pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_log_error
import pandas as pd

print('installing library')
os.system('pip3 install mlbox==0.8.4')

from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *

'''
From the documentation: https://mlbox.readthedocs.io/en/latest/
'''
# install mlblocks

def train_mlbox(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):

	# name model
	modelname=jsonfile[0:-5]+'_mlbox_'+str(default_features).replace("'",'').replace('"','')

	# training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)
	df = {"train" : pd.DataFrame(X_train), "target" : pd.DataFrame(y_train), "test" : pd.DataFrame(X_test)}
	print(df)

	if mtype=='c': 

		# rename files with classification
		modelname=modelname+'_classification'
		model_name=modelname+'.pickle'
		jsonfilename=modelname+'.json'

		# from sklearn.datasets import load_boston
		# dataset = load_boston()
		# df = {"train" : pd.DataFrame(dataset.data), "target" : pd.Series(dataset.target)}
		# print(df['train'][0])
		# print(type(df['train'][0]))
		# data = Drift_thresholder().fit_transform(df)  #deleting non-stable variables

		space = {

				'ne__numerical_strategy' : {"space" : [0, 'mean']},

				'ce__strategy' : {"space" : ["label_encoding", "random_projection", "entity_embedding"]},

				'fs__strategy' : {"space" : ["variance", "rf_feature_importance"]},
				'fs__threshold': {"search" : "choice", "space" : [0.1, 0.2, 0.3]},

				'est__strategy' : {"space" : ["LightGBM"]},
				'est__max_depth' : {"search" : "choice", "space" : [5,6]},
				'est__subsample' : {"search" : "uniform", "space" : [0.6,0.9]}

				}

		best = Optimiser().optimise(space, df, max_evals = 5)
		mse_ =Optimiser().evaluate(best, df)
		pipeline = Predictor().fit_predict(best, df)

		print(best)
		print(mse_)

		# saving model
		print('saving model')
		modelfile=open(model_name,'wb')
		pickle.dump(pipeline, modelfile)
		modelfile.close()

		# SAVE JSON FILE 
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'accuracy':accuracy,
			'model type':'mlblocks_regression',
			'settings': settings,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

	if mtype=='r': 

		# rename files with regression
		modelname=modelname+'_regression'
		model_name=modelname+'.pickle'
		jsonfilename=modelname+'.json'

		params = {"ne__numerical_strategy" : 0,
				"ce__strategy" : "label_encoding",
				"fs__threshold" : 0.1,
				"stck__base_estimators" : [Regressor(strategy="RandomForest"), Regressor(strategy="ExtraTrees")],
				"est__strategy" : "Linear"}

		best = Optimiser().optimise(params, df, max_evals = 5)
		mse_error  =Optimiser().evaluate(best, df)

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
			'model type':'mlblocks_regression',
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