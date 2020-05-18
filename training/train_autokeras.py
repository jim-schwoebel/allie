import os, pickle, json, shutil
print('installing package')
os.system('pip3 install tensorflow==2.1')
os.system('pip3 install autokeras==1.0.2')
import autokeras as ak
import tensorflow as tf
import numpy as np

'''
# plot model
https://autokeras.com/tutorial/structured_data_regression/
'''

def train_autokeras(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

	# create file names
	files=list()
	model_name=common_name_model+'.pickle'

	# remove folder if it exists
	if mtype=='c':
		if 'structured_data_classifier' in os.listdir():
			shutil.rmtree('structured_data_classifier')
		model = ak.StructuredDataClassifier(max_trials=100)
		model.fit(X_train, y_train)
		files.append('structured_data_classifier')
	elif mtype == 'r':
		if 'structured_data_regressor' in os.listdir():
			shutil.rmtree('structured_data_regressor')
		model = ak.StructuredDataRegressor(max_trials=100)
		model.fit(X_train, y_train)
		files.append('structured_data_regressor')

	# show predictions
	predictions=model.predict(X_test).flatten()	
	print(predictions)

	# pickle the model
	picklefile= open(common_name_model+'.pickle','wb')
	pickle.dump(model, picklefile)
	picklefile.close()

	# get variables
	files.append(model_name)
	model_dir=os.getcwd()

	return model_name, model_dir, files
