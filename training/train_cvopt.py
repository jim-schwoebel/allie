import os, sys, shutil
import numpy as np, pandas as pd, scipy as sp
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from cvopt.model_selection import SimpleoptCV
from cvopt.search_setting import search_category, search_numeric


def train_cvopt(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):
	# downloading library
	os.system('pip3 install cvopt==0.4.2')
	os.system('pip3 install scikit-learn==0.22')
	os.system('pip3 install bokeh==1.4.0')

	# create model name
	model_name=common_name_model
	files=list()

	param_distributions = {
		"penalty": search_category(['none', 'l2']),
		"C": search_numeric(0.01, 3.0, "float"), 
		"tol" : search_numeric(0.0001, 0.001, "float"),  
		"class_weight" : search_category([None, "balanced", {0:0.5, 1:0.1}]),
		}

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
						 backend=bk,                     # hyperopt,bayesopt, gaopt or randomopt.
						 )

		opt.fit(X_train, y_train, validation_data=(X_test, y_test))
		ytest_pred = opt.predict(X_test)

		from cvopt.utils import extract_params
		target_index = pd.DataFrame(opt.cv_results_)[pd.DataFrame(opt.cv_results_)["mean_test_score"] == opt.best_score_]["index"].values[0]

		estimator_params, feature_params, feature_select_flag  = extract_params(logdir="./search_usage", 
																				model_id=bk, 
																				target_index=target_index)
																   
		estimator.set_params(**estimator_params)         # Set estimator parameters
		X_train_selected = X_train[:, feature_select_flag] # Extract selected feature columns

		print(estimator)
		print("Train features shape:", X_train.shape)
		print("Train selected features shape:",X_train_selected.shape)

		from cvopt.utils import mk_metafeature
		X_train_meta, X_test_meta = mk_metafeature(X_train, y_train, 
												 logdir="./search_usage", 
												 model_id=bk, 
												 target_index=target_index, 
												 cv=cv, 
												 validation_data=(X_test, y_test), 
												 estimator_method="predict_proba")

		print("Train features shape:", X_train.shape)
		print("Train meta features shape:", X_train_meta.shape)
		print("Test features shape:", X_test.shape)
		print("Test meta features shape:",  X_test_meta.shape)


	os.rename('search_usage', model_name)
	files.append(model_name)
	model_dir=os.getcwd()

	return model_name, model_dir, files