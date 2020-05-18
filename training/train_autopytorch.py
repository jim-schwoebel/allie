import os, json, shutil, pickle, sys
os.system('pip3 install torch==1.5.0')
import torch
import pandas as pd

print('installing library')
os.system('pip3 install autopytorch==0.0.2')

'''
From the documentation: 
--> https://github.com/automl/Auto-PyTorch
# saving/loading torch models: 
--> https://pytorch.org/tutorials/beginner/saving_loading_models.html
'''

def train_autopytorch(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

	# name model
	model_name=common_name_model+'.pickle'
	files=list()

	if mtype=='c': 
		from autoPyTorch import AutoNetClassification
		autonet = AutoNetClassification(log_level='debug', max_runtime=900, min_budget=50, max_budget=150)
		autonet.fit(X_train, y_train, validation_split=0.30)
		print(autonet.predict(X_test).flatten())

	if mtype=='r': 
		from autoPyTorch import AutoNetRegression
		autonet = AutoNetRegression(log_level='debug', max_runtime=900, min_budget=50, max_budget=150)
		autonet.fit(X_train, y_train)
		print(autonet.predict(X_test).flatten())

	print('saving model -->')
	torch.save(autonet, model_name)

	# get model directory
	files.append(model_name)
	files.append('configs.json')
	files.append('results.json')
	model_dir=os.getcwd()

	return model_name, model_dir, files