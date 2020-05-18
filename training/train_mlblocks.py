import os, json, shutil, pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_log_error
print('installing library')
os.system('pip3 install mlprimitives')
os.system('pip3 install mlblocks==0.3.4')
from mlblocks import MLPipeline
# os.system('pip3 install xgboost==0.80')
'''
From the documentation: https://hdi-project.github.io/MLBlocks/pipeline_examples/single_table.html
'''
def train_mlblocks(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

	# name model
	model_name=common_name_model+'.pickle'
	files=list()

	if mtype=='c': 

		primitives = ['sklearn.impute.SimpleImputer',
					  'xgboost.XGBClassifier']

		init_params = {'sklearn.impute.SimpleImputer': {'strategy': 'median'},
		  			    'xgboost.XGBClassifier': {'learning_rate': 0.1}}

		pipeline = MLPipeline(primitives, init_params=init_params)

		pipeline.fit(X_train, y_train)

		# saving model
		print('saving model')
		modelfile=open(model_name,'wb')
		pickle.dump(pipeline, modelfile)
		modelfile.close()


	if mtype=='r': 

		primitives = ['sklearn.impute.SimpleImputer',
					  'xgboost.XGBRegressor']
		pipeline = MLPipeline(primitives)
		pipeline.fit(X_train, y_train)

	# get model directory
	files.append(model_name)
	model_dir=os.getcwd()

	return model_name, model_dir, files