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

def train_neuraxle(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

	# get train and test data 
	model_name=common_name_model+'.pickle'
	files=list()

	if mtype in ['classification', 'c']:
		print('neuraxle currently does not support classsification...')

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

		# export pickle file 
		print('saving model - %s'%(model_name))
		f=open(model_name,'wb')
		pickle.dump(pipeline, f)
		f.close()

		files.append(model_name)
	
	model_dir=os.getcwd()

	return model_name, model_dir, files