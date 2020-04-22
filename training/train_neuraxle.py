import time
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pickle, json, os, shutil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

def train_neuraxle(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):
	# install library
	os.system('pip3 install neuraxle==0.4.0')
	from neuraxle.pipeline import Pipeline
	from neuraxle.steps.numpy import NumpyShapePrinter
	from neuraxle.steps.sklearn import RidgeModelStacking
	from neuraxle.union import AddFeatures
	from neuraxle.checkpoints import DefaultCheckpoint
	from neuraxle.hyperparams.distributions import RandInt
	from neuraxle.hyperparams.space import HyperparameterSpace
	from neuraxle.metaopt.auto_ml import RandomSearchHyperparameterSelectionStrategy
	from neuraxle.metaopt.callbacks import MetricCallback, ScoringCallback
	from neuraxle.pipeline import ResumablePipeline, DEFAULT_CACHE_FOLDER, Pipeline
	from neuraxle.steps.flow import ExpandDim
	from neuraxle.steps.loop import ForEachDataInput
	from neuraxle.steps.misc import Sleep
	from neuraxle.steps.numpy import MultiplyByN
	from neuraxle.steps.numpy import NumpyShapePrinter
	from neuraxle.union import AddFeatures
	
	# get train and test data 
	X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)
	modelname=jsonfile[0:-5]+'_'+str(default_features).replace("'",'').replace('"','')+'_neuraxle'

	if mtype in ['classification', 'c']:
		print('neuraxle currently does not support training classification models. We are working on this soon')
		print('----> please use another model training script')
		model_name=''
		model_dir=os.getcwd()

	elif mtype in ['regression', 'r']:

		p = Pipeline([
			NumpyShapePrinter(),
			AddFeatures([
				PCA(n_components=2),
				FastICA(n_components=2),
			]),
			NumpyShapePrinter(),
			RidgeModelStacking([
				GradientBoostingRegressor(),
				GradientBoostingRegressor(n_estimators=500),
				GradientBoostingRegressor(max_depth=5),
				KMeans(),
			]),
			NumpyShapePrinter(),
		])

		# Fitting and evaluating the pipeline.
		# X_train data shape: (batch, different_lengths, n_feature_columns)
		# y_train data shape: (batch, different_lengths)
		pipeline = p.fit(X_train, y_train)
		y_test_predicted = pipeline.predict(X_test)
		r2score = r2_score(y_test_predicted, y_test)

		print('------R2SCORE-------')
		print(r2score)

		# export pickle file 
		print('saving model - %s'%(modelname+'.pickle'))
		f=open(modelname+'.pickle','wb')
		pickle.dump(pipeline, f)
		f.close()

		jsonfilename='%s.json'%(modelname)
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'r2_score':r2score,
			'model type':'neuraxle_regression',
			'settings': settings,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

		cur_dir2=os.getcwd()
		try:
			os.chdir(problemtype+'_models')
		except:
			os.mkdir(problemtype+'_models')
			os.chdir(problemtype+'_models')

		# now move all the files over to proper model directory 
		shutil.move(cur_dir2+'/'+jsonfilename, os.getcwd()+'/'+jsonfilename)
		shutil.move(cur_dir2+'/'+jsonfilename[0:-5]+'.pickle', os.getcwd()+'/'+jsonfilename[0:-5]+'.pickle')

		# get model_name 
		model_name=jsonfilename[0:-5]+'.pickle'
		model_dir=os.getcwd()
	
	return model_name, model_dir