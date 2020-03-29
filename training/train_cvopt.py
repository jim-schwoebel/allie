import os, sys, shutil
import numpy as np, pandas as pd, scipy as sp
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from cvopt.model_selection import SimpleoptCV
from cvopt.search_setting import search_category, search_numeric


def train_cvopt(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):
	# training and testing sets
	Xtrain, Xtest, ytrain, ytest = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)
	modelname=jsonfile[0:-5]+'_cvopt_'+str(default_features).replace("'",'').replace('"','')
	model_name=modelname+'.pickle'

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

		opt.fit(Xtrain, ytrain, validation_data=(Xtest, ytest))
		ytest_pred = opt.predict(Xtest)


		from cvopt.utils import extract_params
		target_index = pd.DataFrame(opt.cv_results_)[pd.DataFrame(opt.cv_results_)["mean_test_score"] == opt.best_score_]["index"].values[0]

		estimator_params, feature_params, feature_select_flag  = extract_params(logdir="./search_usage", 
																				model_id=bk, 
																				target_index=target_index)
																   
		estimator.set_params(**estimator_params)         # Set estimator parameters
		Xtrain_selected = Xtrain[:, feature_select_flag] # Extract selected feature columns

		print(estimator)
		print("Train features shape:", Xtrain.shape)
		print("Train selected features shape:",Xtrain_selected.shape)

		from cvopt.utils import mk_metafeature
		Xtrain_meta, Xtest_meta = mk_metafeature(Xtrain, ytrain, 
												 logdir="./search_usage", 
												 model_id=bk, 
												 target_index=target_index, 
												 cv=cv, 
												 validation_data=(Xtest, ytest), 
												 estimator_method="predict_proba")

		print("Train features shape:", Xtrain.shape)
		print("Train meta features shape:", Xtrain_meta.shape)
		print("Test features shape:", Xtest.shape)
		print("Test meta features shape:",  Xtest_meta.shape)

	cur_dir2=os.getcwd()
	os.rename('search_usage', modelname)

	try:
		os.chdir(problemtype+'_models')
	except:
		os.mkdir(problemtype+'_models')
		os.chdir(problemtype+'_models')

	# now move all the files over to proper model directory 
	shutil.copytree(cur_dir2+'/'+modelname, os.getcwd()+'/'+modelname)
	shutil.rmtree(cur_dir2+'/'+modelname)

	model_dir=os.getcwd()

	return model_name, model_dir