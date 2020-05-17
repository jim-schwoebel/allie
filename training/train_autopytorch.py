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
	modelname=common_name_model+'.pickle'

	if mtype=='c': 
		from autoPyTorch import AutoNetClassification

		# running Auto-PyTorch
		autonet = AutoNetClassification("tiny_cs",  # config preset
		                                    log_level='info',
		                                    max_runtime=600,
		                                    min_budget=30,
		                                    max_budget=90)

		autonet.fit(X_train, y_train, validation_split=0.30)

		pytorch_model = autonet.get_pytorch_model()
		print(pytorch_model)
		# saving model
		print('saving model')
		torch.save(pytorch_model, model_name)

	if mtype=='r': 

		# run model session
		from autoPyTorch import AutoNetRegression

		# Note: every parameter has a default value, you do not have to specify anything. The given parameter allow a fast test.
		autonet = AutoNetRegression(budget_type='epochs', min_budget=1, max_budget=9, num_iterations=1, log_level='info')
		pipeline = autonet.fit(X_train, y_train)

		# saving model
		print('saving model')
		modelfile=open(model_name,'wb')
		pickle.dump(pipeline, modelfile)
		modelfile.close()

	# get model directory
	files.append(modelfile)
	model_dir=os.getcwd()

	return model_name, model_dir, files