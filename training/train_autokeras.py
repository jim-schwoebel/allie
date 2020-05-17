import os, sys, pickle, json, random, shutil, time, yaml
print('installing package')
os.system('pip3 install tf-nightly==2.3.0.dev20200516')
os.system('pip3 install autokeras==1.0.2')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
import pandas as pd
import time

'''

Saving and loading autokeras models 

for autokeras 1.0:
save the model
model.save('my_model')

load the model
import tensorflow as tf
new_model = tf.keras.models.load_model('my_model')

'''

def train_autokeras(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

	import autokeras as ak
	import tensorflow as tf

	# create file names
	files=list()
	model_name=common_name_model+'.h5'

	if mtype=='c':
		# Initialize the classifier.
		# 30 trials later
		clf = ak.StructuredDataClassifier(max_trials=1)
		clf.fit(X_train, y_train)

		# Evaluate the accuracy of the found model.
		predictions=clf.predict(X_test)	
		accuracy=accuracy_score(y_test, predictions)
		print('------------------------------------')
		print('	BEST MODEL 	 			')
		print('------------------------------------')
		print('ACCURACY')
		print(accuracy)
		time.sleep(3)
		# saving model
		# debugged from stackoverflow: https://stackoverflow.com/questions/59533584/how-to-save-load-models-in-autokeras-1-0
		print('saving model architecture')
		best_model = clf.tuner.get_best_model()
		#model = best_model.export_model()

		print(best_model)
		print(type(best_model))
		print(model_name)
		time.sleep(5)
		tf.keras.models.save_model(best_model, model_name, save_format='h5')
		json_string = best_model.to_json()
		print(json_string)

	elif mtype == 'r':

		# change to 100 trials 
		train_dataset=pd.read_csv(trainfile)
		regressor = ak.StructuredDataRegressor(max_trials=1)
		regressor.fit(X_train, y_train)
		# Evaluate the accuracy of the found model.
		predictions=regressor.predict(X_test)		
		mse_error=mean_squared_error(y_test, predictions)
		print('------------------------------------')
		print('	BEST MODEL 	 			')
		print('------------------------------------')
		print('MSE-ERROR')
		print(mse_error)
		time.sleep(3)
		# saving model
		# debugged from stackoverflow: https://stackoverflow.com/questions/59533584/how-to-save-load-models-in-autokeras-1-0
		print('saving model architecture')
		best_model = regressor.tuner.get_best_model()
		model = best_model.export_model()
		print(best_model)
		print(type(best_model))
		print(model_name)
		time.sleep(5)
		model.save(model_name)
		json_string = best_model.to_json()
		print(json_string)

	# pickle the model
	picklefile= open(common_name_model+'.pickle','wb')
	pickle.dump(model, picklefile)
	picklefile.close()

	# tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
	
	# get variables
	files.append(model_name+'.h5')
	files.append('structured_data_classifier')
	model_dir=os.getcwd()

	return model_name, model_dir, files
