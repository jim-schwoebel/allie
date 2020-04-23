import os, json, shutil
from sklearn.model_selection import train_test_split
'''
From the documentation: https://hdi-project.github.io/MLBlocks/pipeline_examples/single_table.html
'''
# install mlblocks

def train_mlblocks(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):

	print('installing library')
	os.system('pip3 install mlblocks==0.3.4')

	# name model
	modelname=jsonfile[0:-5]+'_mlblocks_'+str(default_features).replace("'",'').replace('"','')
	model_name=modelname+'.pickle'
	jsonfilename=modelname+'.json'

	# training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)

	if mtype=='classification':

		from mlblocks import MLPipeline

		primitives = [
			'sklearn.preprocessing.StandardScaler',
			'xgboost.XGBClassifier'
		]
		init_params = {
			'xgboost.XGBClassifier': {
				'learning_rate': 0.1
			}
		}
		pipeline = MLPipeline(primitives, init_params)
		pipeline.fit(X_train, y_train)
		predictions = pipeline.predict(X_test)
		accuracy=dataset.score(y_test, predictions)

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
			'model type':'mlblocks_classification',
			'settings': settings,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

	if mtype=='regression':

		from mlblocks import MLPipeline

		primitives = [
			'sklearn.ensemble.RandomForestRegressor'
		]
		pipeline = MLPipeline(primitives)
		pipeline.fit(X_train, y_train)
		predictions = pipeline.predict(X_test)

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
	os.remove(curdir2+'/'+jsonfilename)

	# get model directory
	model_dir=os.getcwd()

	return model_name, model_dir