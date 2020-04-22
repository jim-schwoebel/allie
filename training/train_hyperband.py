import os, sys, shutil, pickle, json
from scipy.stats import randint as sp_randint
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def train_hyperband(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):
	# install
	os.system('pip3 install scikit-hyperband==0.0.1')
	from hyperband import HyperbandSearchCV
	
	# training and testing sets
	X, X_test, y, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)

	if mtype in ['classification', 'c']:

		modelname=jsonfile[0:-5]+'_hyperband_classification_'+str(default_features).replace("'",'').replace('"','')
		model_name=modelname+'.pickle'
		model_transformer=modelname+'_transformer.pickle'
		jsonfilename=modelname+'.json'

		model = RandomForestClassifier()
		param_dist = {
		    'max_depth': [3, None],
		    'max_features': sp_randint(1, 11),
		    'min_samples_split': sp_randint(2, 11),
		    'min_samples_leaf': sp_randint(1, 11),
		    'bootstrap': [True, False],
		    'criterion': ['gini', 'entropy']
		}

		transformer=LabelBinarizer().fit(y)
		y_new = transformer.transform(y)

		search = HyperbandSearchCV(model, param_dist, 
		                           resource_param='n_estimators',
		                           scoring='roc_auc')
		search.fit(X, y_new)
		params=search.best_params_
		print('-----')
		print('best params: ')
		print(params)
		print('------')
		accuracy=search.score(X_test, transformer.transform(y_test))

		# SAVE ML MODEL
		modelfile=open(model_name,'wb')
		pickle.dump(search, modelfile)
		modelfile.close()

		# SAVE TRANSFORMER 
		modelfile=open(model_transformer, 'wb')
		pickle.dump(transformer, modelfile)
		modelfile.close()

		# SAVE JSON FILE 
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':model_name,
			'accuracy':accuracy,
			'model type':'hyperband_randomforest_classification',
			'settings': settings,
			'transformer': model_transformer,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

		# move model to proper directory
		cur_dir2=os.getcwd()
		try:
			os.chdir(problemtype+'_models')
		except:
			os.mkdir(problemtype+'_models')
			os.chdir(problemtype+'_models')

		# now move all the files over to proper model directory 
		shutil.move(cur_dir2+'/'+jsonfilename, os.getcwd()+'/'+jsonfilename)
		shutil.move(cur_dir2+'/'+model_name, os.getcwd()+'/'+model_name)
		shutil.move(cur_dir2+'/'+model_transformer, os.getcwd()+'/'+model_transformer)

	elif mtype in ['regression', 'r']:

		print('hyperband currently does not support regression modeling.')
		model_name=''

	model_dir=os.getcwd()

	return model_name, model_dir