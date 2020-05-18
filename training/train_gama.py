import os, sys, shutil, pickle, json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
# install library
print('installing library')
os.system('pip3 install gama==20.1.0')
from gama import GamaClassifier, GamaRegressor

def train_gama(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

	model_name=common_name_model+'.pickle'
	files=list()

	if mtype in ['c']:

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

	elif mtype in ['regression', 'r']:

		automl = GamaRegressor(max_total_time=180, keep_analysis_log=None, n_jobs=1)
		print("Starting GAMA `fit` - usually takes around 3 minutes but can take longer for large datasets")
		automl.fit(X_train, y_train)

		predictions = automl.predict(X_test)
		mse_error=mean_squared_error(y_test, predictions)
		print("MSE:", mse_error)

	# SAVE ML MODEL
	modelfile=open(model_name,'wb')
	pickle.dump(automl, modelfile)
	modelfile.close()

	files.append(model_name)
	model_dir=os.getcwd()

	return model_name, model_dir, files
