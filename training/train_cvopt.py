import os, sys, shutil, pickle
# downloading library
os.system('pip3 install cvopt==0.4.3')
# os.system('pip3 install scikit-learn==0.22')
os.system('pip3 install bokeh==1.4.0')
os.system('pip3 install pandas==1.0.3')
os.system('pip3 install numpy==1.17')
import numpy as np
import pandas as pd
import scipy as sp
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from cvopt.model_selection import SimpleoptCV
from cvopt.search_setting import search_category, search_numeric
from sklearn.externals import joblib


def train_cvopt(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

	# create model name
	model_name=common_name_model+'.pickle'
	files=list()
	
	param_distributions = {
		"penalty": search_category(['none', 'l2']),
		"C": search_numeric(0.01, 3.0, "float"), 
		"tol" : search_numeric(0.0001, 0.001, "float"),  
		"class_weight" : search_category([None, "balanced", {0:0.5, 1:0.1}]),
		}

	# delete search_usage directory
	if 'search_usage' in os.listdir():
		shutil.rmtree('search_usage')
		
	for bk in ["hyperopt", "gaopt", "bayesopt", "randomopt"]:
		estimator = LogisticRegression()
		cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

		opt = SimpleoptCV(estimator, param_distributions, 
						 scoring="roc_auc",              # Objective of search
						 cv=cv,                          # Cross validation setting
						 max_iter=32,                    # Number of search
						 n_jobs=3,                       # Number of jobs to run in parallel.
						 verbose=2,                      # 0: don't display status, 1:display status by stdout, 2:display status by graph 
						 logdir="./search_usage",        # If this path is specified, save the log.
						 model_id=bk,                    # used estimator's dir and file name in save.
						 save_estimator=2,               # estimator save setting.
						 backend=bk)                     # hyperopt,bayesopt, gaopt or randomopt.
						 

		opt.fit(X_train, y_train, validation_data=(X_test, y_test))

		from cvopt.utils import extract_params
		target_index = pd.DataFrame(opt.cv_results_)[pd.DataFrame(opt.cv_results_)["mean_test_score"] == opt.best_score_]["index"].values[0]
		estimator_params, feature_params, feature_select_flag  = extract_params(logdir="./search_usage", 
																				model_id=bk, 
																				target_index=target_index)
																   
		estimator.set_params(**estimator_params)         # Set estimator parameters
		print(estimator)
		estimator.fit(X_train, y_train)

		X_train_selected = X_train[:, feature_select_flag] # Extract selected feature columns

		
		print("Train features shape:", X_train.shape)
		print("Train selected features shape:",X_train_selected.shape)

		picklefile=open(bk+'.pickle','wb')
		pickle.dump(estimator,picklefile)
		picklefile.close()

		# from cvopt.utils import mk_metafeature
		# X_train_meta, X_test_meta = mk_metafeature(X_train, y_train, 
		# 										 logdir="./search_usage", 
		# 										 model_id=bk, 
		# 										 target_index=target_index, 
		# 										 cv=cv, 
		# 										 validation_data=(X_test, y_test), 
		# 										 estimator_method="predict_proba")

		# print("Train features shape:", X_train.shape)
		# print("Train meta features shape:", X_train_meta.shape)
		# print("Test features shape:", X_test.shape)
		# print("Test meta features shape:",  X_test_meta.shape)

	# now export the best model in terms of accuracy
	curdir=os.getcwd()
	os.chdir('search_usage')
	os.chdir('cv_results')

	# load csv docs
	bayesopt=pd.read_csv('bayesopt.csv')
	gaopt=pd.read_csv('gaopt.csv')
	hyperopt=pd.read_csv('hyperopt.csv')
	randomopt=pd.read_csv('randomopt.csv')

	# get max values per types of array
	bayesopt_=np.amax(np.array(bayesopt['mean_test_score']))
	gaopt_=np.amax(np.array(gaopt['mean_test_score']))
	hyperopt_=np.amax(np.array(hyperopt['mean_test_score']))
	randomopt_=np.amax(np.array(randomopt['mean_test_score']))

	# get total groups
	total=np.array([bayesopt_, gaopt_, hyperopt_, randomopt_])
	totalmax=np.amax(total)

	os.chdir(curdir)

	if bayesopt_ == totalmax:
		os.rename('bayesopt.pickle',model_name)
		os.remove('randomopt.pickle')
		os.remove('hyperopt.pickle')
		os.remove('gaopt.pickle')
	elif gaopt_ == totalmax:
		os.rename('gaopt.pickle',model_name)
		os.remove('randomopt.pickle')
		os.remove('hyperopt.pickle')
		os.remove('bayesopt.pickle')
	elif hyperopt_ == totalmax:
		os.rename('hyperopt.pickle',model_name)
		os.remove('randomopt.pickle')
		os.remove('gaopt.pickle')
		os.remove('bayesopt.pickle')
	elif randomopt_ == totalmax:
		os.rename('randomopt.pickle',model_name)
		os.remove('hyperopt.pickle')
		os.remove('gaopt.pickle')
		os.remove('bayesopt.pickle')

	# now add all the relevant files to copy over
	files.append(model_name)
	files.append('search_usage')
	files.append('model.html')
	model_dir=os.getcwd()

	return model_name, model_dir, files