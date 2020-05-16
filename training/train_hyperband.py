import os, pickle
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier

def prev_dir(directory):
	g=directory.split('/')
	dir_=''
	for i in range(len(g)):
		if i != len(g)-1:
			if i==0:
				dir_=dir_+g[i]
			else:
				dir_=dir_+'/'+g[i]
	# print(dir_)
	return dir_

def train_hyperband(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):
	# install
	curdir=os.getcwd()
	os.chdir(prev_dir(os.getcwd())+'/training/helpers/hyperband')
	os.system('python3 setup.py install')
	from hyperband import HyperbandSearchCV
	os.chdir(curdir)
	
	# training and testing sets
	files=list()
	model_name=common_name_model+'.pickle'

	if mtype in ['classification', 'c']:

		model = RandomForestClassifier()
		param_dist = {
		    'max_depth': [3, None],
		    'max_features': sp_randint(1, 11),
		    'min_samples_split': sp_randint(2, 11),
		    'min_samples_leaf': sp_randint(1, 11),
		    'bootstrap': [True, False],
		    'criterion': ['gini', 'entropy']
		}

		search = HyperbandSearchCV(model, param_dist, 
		                           resource_param='n_estimators',
		                           scoring='roc_auc')
		search.fit(X_train, y_train)
		params=search.best_params_
		print('-----')
		print('best params: ')
		print(params)
		print('------')
		accuracy=search.score(X_test, y_test)

		# SAVE ML MODEL
		modelfile=open(model_name,'wb')
		pickle.dump(search, modelfile)
		modelfile.close()

	elif mtype in ['regression', 'r']:

		print('hyperband currently does not support regression modeling.')
		model_name=''

	model_dir=os.getcwd()
	files.append(model_name)

	return model_name, model_dir, files