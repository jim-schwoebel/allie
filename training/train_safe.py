import os, sys, shutil, pickle, json
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from SafeTransformer import SafeTransformer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_safe(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):
	# training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)
	X_train=pd.DataFrame(X_train)
	X_test=pd.DataFrame(X_test)
	y_train=pd.DataFrame(y_train)
	y_test=pd.DataFrame(y_test)

	if mtype in ['classification', 'c']:

		modelname=jsonfile[0:-5]+'_safe_classification_'+str(default_features).replace("'",'').replace('"','')
		model_name=modelname+'.pickle'
		model_name_sur = modelname+'_surrogate.pickle'
		model_name_base = modelname+'_base.pickle'

		jsonfilename=modelname+'.json'

		print("Starting FIT")
		surrogate_model = XGBClassifier().fit(X_train, y_train)

		base_model = LogisticRegression().fit(X_train, y_train)
		base_predictions = base_model.predict(X_test)

		# transform data
		pen=1
		safe_transformer = SafeTransformer(model=surrogate_model, penalty=pen)
		safe_transformer = safe_transformer.fit(X_train)
		X_train_transformed = safe_transformer.transform(X_train)
		X_test_transformed = safe_transformer.transform(X_test)

		# surrogate model 
		model_transformed = LogisticRegression()
		model_transformed = model_transformed.fit(X_train_transformed, y_train)
		surrogate_predictions = model_transformed.predict(X_test_transformed)

		acc_surrogate=accuracy_score(y_test, surrogate_predictions)
		acc_base=accuracy_score(y_test, base_predictions)
		acc_model=accuracy_score(y_test, surrogate_model.predict(X_test))

		print(acc_surrogate)
		print(acc_base)
		print(acc_model)

		# SAVE ML MODEL
		modelfile=open(model_name,'wb')
		pickle.dump(surrogate_model, modelfile)
		modelfile.close()

		# SAVE SURROGATE ML MODEL
		modelfile=open(model_name_sur,'wb')
		pickle.dump(safe_transformer, modelfile)
		modelfile.close()

		# SAVE TRANSFORMED MODEL 
		modelfile=open(model_name_base, 'wb')
		pickle.dump(base_model, modelfile)
		modelfile.close()

		# SAVE JSON FILE 
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'models':{model_name: acc_model, 
					 model_name_sur: acc_surrogate, 
					 model_Name_base: acc_base}
			'model type':'safe_classification',
			'settings': settings,
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
		shutil.move(cur_dir2+'/'+model_name_sur, os.getcwd()+'/'+model_name_sur)
		shutil.move(cur_dir2+'/'+model_name, os.getcwd()+'/'+model_name)
		shutil.move(cur_dir2+'/'+model_name_base, os.getcwd()+'/'+model_name_base)

	elif mtype in ['regression', 'r']:

		modelname=jsonfile[0:-5]+'_safe_regression_'+str(default_features).replace("'",'').replace('"','')
		model_name=modelname+'.pickle'
		jsonfilename=modelname+'.json'

		surrogate_model = GradientBoostingRegressor(n_estimators=100,
		    max_depth=4,
		    learning_rate=0.1,
		    loss='huber')
		surrogate_model = surrogate_model.fit(X_train, y_train)

		linear_model = LinearRegression()
		safe_transformer = SafeTransformer(surrogate_model, penalty = 0.84)
		pipe = Pipeline(steps=[('safe', safe_transformer), ('linear', linear_model)])
		pipe = pipe.fit(X_train, y_train)
		predictions = pipe.predict(X_test)
		mse_error=mean_squared_error(y_test, predictions)

		print("MSE:", mse_error)

		# SAVE TRANSFORMED MODEL 
		modelfile=open(model_name, 'wb')
		pickle.dump(pipe, modelfile)
		modelfile.close()

		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':model_name,
			'mse_error':mse_error,
			'model type':'safe_regression',
			'settings': settings,
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

	model_dir=os.getcwd()

	return model_name, model_dir