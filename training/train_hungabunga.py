'''
               AAA               lllllll lllllll   iiii                      
              A:::A              l:::::l l:::::l  i::::i                     
             A:::::A             l:::::l l:::::l   iiii                      
            A:::::::A            l:::::l l:::::l                             
           A:::::::::A            l::::l  l::::l iiiiiii     eeeeeeeeeeee    
          A:::::A:::::A           l::::l  l::::l i:::::i   ee::::::::::::ee  
         A:::::A A:::::A          l::::l  l::::l  i::::i  e::::::eeeee:::::ee
        A:::::A   A:::::A         l::::l  l::::l  i::::i e::::::e     e:::::e
       A:::::A     A:::::A        l::::l  l::::l  i::::i e:::::::eeeee::::::e
      A:::::AAAAAAAAA:::::A       l::::l  l::::l  i::::i e:::::::::::::::::e 
     A:::::::::::::::::::::A      l::::l  l::::l  i::::i e::::::eeeeeeeeeee  
    A:::::AAAAAAAAAAAAA:::::A     l::::l  l::::l  i::::i e:::::::e           
   A:::::A             A:::::A   l::::::ll::::::li::::::ie::::::::e          
  A:::::A               A:::::A  l::::::ll::::::li::::::i e::::::::eeeeeeee  
 A:::::A                 A:::::A l::::::ll::::::li::::::i  ee:::::::::::::e  
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee  
                                                                             
|  \/  |         | |    | |  / _ \ | ___ \_   _|
| .  . | ___   __| | ___| | / /_\ \| |_/ / | |  
| |\/| |/ _ \ / _` |/ _ \ | |  _  ||  __/  | |  
| |  | | (_) | (_| |  __/ | | | | || |    _| |_ 
\_|  |_/\___/ \__,_|\___|_| \_| |_/\_|    \___/ 

Train models using hungabunga: https://github.com/ypeleg/HungaBunga

This is enabled if the default_training_script = ['hungabunga']
'''
import os, sys, pickle, json, random, shutil, time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, Matern, StationaryKernelMixin, WhiteKernel
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler


############################
### CORE
###########################

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', 'Solver terminated early.*')

import sklearn.model_selection
import numpy as np
nan = float('nan')
import traceback

from pprint import pprint
from collections import Counter
from multiprocessing import cpu_count
from time import time
from tabulate import tabulate
try: from tqdm import tqdm
except: tqdm = lambda x: x

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit as sss, ShuffleSplit as ss, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import model_selection


TREE_N_ENSEMBLE_MODELS = [RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, DecisionTreeRegressor,ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor]


class GridSearchCVProgressBar(sklearn.model_selection.GridSearchCV):
		def _get_param_iterator(self):
				iterator = super(GridSearchCVProgressBar, self)._get_param_iterator()
				iterator = list(iterator)
				n_candidates = len(iterator)
				cv = sklearn.model_selection._split.check_cv(self.cv, None)
				n_splits = getattr(cv, 'n_splits', 3)
				max_value = n_candidates * n_splits
				class ParallelProgressBar(sklearn.model_selection._search.Parallel):
						def __call__(self, iterable):
								iterable = tqdm(iterable, total=max_value)
								iterable.set_description("GridSearchCV")
								return super(ParallelProgressBar, self).__call__(iterable)
				sklearn.model_selection._search.Parallel = ParallelProgressBar
				return iterator


class RandomizedSearchCVProgressBar(sklearn.model_selection.RandomizedSearchCV):
		def _get_param_iterator(self):
				iterator = super(RandomizedSearchCVProgressBar, self)._get_param_iterator()
				iterator = list(iterator)
				n_candidates = len(iterator)
				cv = sklearn.model_selection._split.check_cv(self.cv, None)
				n_splits = getattr(cv, 'n_splits', 3)
				max_value = n_candidates * n_splits
				class ParallelProgressBar(sklearn.model_selection._search.Parallel):
						def __call__(self, iterable):
								iterable = tqdm(iterable, total=max_value)
								iterable.set_description("RandomizedSearchCV")
								return super(ParallelProgressBar, self).__call__(iterable)
				sklearn.model_selection._search.Parallel = ParallelProgressBar
				return iterator


def upsample_indices_clf(inds, y):
		assert len(inds) == len(y)
		countByClass = dict(Counter(y))
		maxCount = max(countByClass.values())
		extras = []
		for klass, count in countByClass.items():
				if maxCount == count: continue
				ratio = int(maxCount / count)
				cur_inds = inds[y == klass]
				extras.append(np.concatenate( (np.repeat(cur_inds, ratio - 1), np.random.choice(cur_inds, maxCount - ratio * count, replace=False))))
		return np.concatenate([inds] + extras)


def cv_clf(x, y, test_size = 0.2, n_splits = 5, random_state=None, doesUpsample = True):
		sss_obj = sss(n_splits, test_size, random_state=random_state).split(x, y)
		if not doesUpsample: yield sss_obj
		for train_inds, valid_inds in sss_obj: yield (upsample_indices_clf(train_inds, y[train_inds]), valid_inds)


def cv_reg(x, test_size = 0.2, n_splits = 5, random_state=None): return ss(n_splits, test_size, random_state=random_state).split(x)


def timeit(klass, params, x, y):
		start = time()
		clf = klass(**params)
		clf.fit(x, y)
		return time() - start


def main_loop(models_n_params, x, y, isClassification, test_size = 0.2, n_splits = 5, random_state=None, upsample=True, scoring=None, verbose=True, n_jobs =cpu_count() - 1, brain=False, grid_search=True):
		def cv_(): return cv_clf(x, y, test_size, n_splits, random_state, upsample) if isClassification else cv_reg(x, test_size, n_splits, random_state)
		res = []
		num_features = x.shape[1]
		scoring = scoring or ('accuracy' if isClassification else 'neg_mean_squared_error')
		if brain: print('Scoring criteria:', scoring)
		for i, (clf_Klass, parameters) in enumerate(tqdm(models_n_params)):
				try:
						if brain: print('-'*15, 'model %d/%d' % (i+1, len(models_n_params)), '-'*15)
						if brain: print(clf_Klass.__name__)
						if clf_Klass == KMeans: parameters['n_clusters'] = [len(np.unique(y))]
						elif clf_Klass in TREE_N_ENSEMBLE_MODELS: parameters['max_features'] = [v for v in parameters['max_features'] if v is None or type(v)==str or v<=num_features]
						if grid_search: clf_search = GridSearchCVProgressBar(clf_Klass(), parameters, scoring, cv=cv_(), n_jobs=n_jobs)
						else: clf_search = RandomizedSearchCVProgressBar(clf_Klass(), parameters, scoring, cv=cv_(), n_jobs=n_jobs)
						clf_search.fit(x, y)
						timespent = timeit(clf_Klass, clf_search.best_params_, x, y)
						if brain: print('best score:', clf_search.best_score_, 'time/clf: %0.3f seconds' % timespent)
						if brain: print('best params:')
						if brain: pprint(clf_search.best_params_)
						if verbose:
								print('validation scores:', clf_search.cv_results_['mean_test_score'])
								print('training scores:', clf_search.cv_results_['mean_train_score'])
						res.append((clf_search.best_estimator_, clf_search.best_score_, timespent))
				except Exception as e:
						if verbose: traceback.print_exc()
						res.append((clf_Klass(), -np.inf, np.inf))
		if brain: print('='*60)
		if brain: print(tabulate([[m.__class__.__name__, '%.3f'%s, '%.3f'%t] for m, s, t in res], headers=['Model', scoring, 'Time/clf (s)']))
		winner_ind = np.argmax([v[1] for v in res])
		winner = res[winner_ind][0]
		if brain: print('='*60)
		if brain: print('The winner is: %s with score %0.3f.' % (winner.__class__.__name__, res[winner_ind][1]))
		return winner, res



if __name__ == '__main__':
		y = np.array([0,1,0,0,0,3,1,1,3])
		x = np.zeros(len(y))
		for t, v in cv_reg(x): print(v,t)
		for t, v in cv_clf(x, y, test_size=5): print(v,t)

###############
### PARAMS
#################

import numpy as np

penalty_12 = ['l1', 'l2']
penalty_12none = ['l1', 'l2', None]
penalty_12e = ['l1', 'l2', 'elasticnet']
penalty_all = ['l1', 'l2', None, 'elasticnet']
max_iter = [100, 300, 1000]
max_iter_2 = [25]
max_iter_inf = [100, 300, 500, 1000, np.inf]
max_iter_inf2 = [100, 300, 500, 1000, -1]
tol = [1e-4, 1e-3, 1e-2]
warm_start = [True, False]
alpha = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 3, 10]
alpha_small = [1e-5, 1e-3, 0.1, 1]
n_iter = [5, 10, 20]
eta0 = [1e-4, 1e-3, 1e-2, 0.1]
C = [1e-2, 0.1, 1, 5, 10]
C_small = [ 0.1, 1, 5]
epsilon = [1e-3, 1e-2, 0.1, 0]
normalize =  [True, False]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
degree = [1, 2, 3, 4, 5]
gamma = list(np.logspace(-9, 3, 6)) + ['auto']
gamma_small = list(np.logspace(-6, 3, 3)) + ['auto']
coef0 = [0, 0.1, 0.3, 0.5, 0.7, 1]
coef0_small = [0, 0.4, 0.7, 1]
shrinking = [True, False]
nu = [1e-4, 1e-2, 0.1, 0.3, 0.5, 0.75, 0.9]
nu_small = [1e-2, 0.1, 0.5, 0.9]
n_neighbors = [5, 7, 10, 15, 20]
neighbor_algo = ['ball_tree', 'kd_tree', 'brute']
neighbor_leaf_size = [1, 2, 5, 10, 20, 30, 50, 100]
neighbor_metric = ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']
neighbor_radius = [1e-2, 0.1, 1, 5, 10]
learning_rate = ['constant', 'invscaling', 'adaptive']
learning_rate_small = ['invscaling', 'adaptive']
learning_rate2=[0.05, 0.10, 0.15]
n_estimators = [2, 3, 5, 10, 25, 50, 100]
n_estimators_small = [2, 10, 25, 100]
max_features = [3, 5, 10, 25, 50, 'auto', 'log2', None]
max_features_small = [3, 5, 10, 'auto', 'log2', None]
max_depth = [None, 3, 5, 7, 10]
max_depth_small =  [None, 5, 10]
min_samples_split = [2, 5, 10, 0.1]
min_impurity_split = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
tree_learning_rate = [0.8, 1]
min_samples_leaf = [2]

# for regression

import warnings
warnings.filterwarnings('ignore')
from multiprocessing import cpu_count

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor, PassiveAggressiveRegressor, RANSACRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier

def train_hungabunga(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

		model_name=common_name_model+'.pickle'
		files=list()

		if mtype in [' classification', 'c']:

				linear_models_n_params = [
						(SGDClassifier,
						 {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
							'alpha': [0.0001, 0.001, 0.1],
							'penalty': penalty_12none, 
							'max_iter':max_iter}),

						(LogisticRegression,
						 {'penalty': penalty_12, 'max_iter': max_iter, 'tol': tol,  'warm_start': warm_start, 'C':C, 'solver': ['liblinear']}),

						(Perceptron,
						 {'penalty': penalty_all, 
						 'alpha': alpha, 
						 'n_iter': n_iter, 
						 'eta0': eta0, 
						 'warm_start': warm_start}),

						(PassiveAggressiveClassifier,
						 {'C': C, 'n_iter': n_iter, 
						 'warm_start': warm_start,
						 'loss': ['hinge', 'squared_hinge']})]

				linear_models_n_params_small = linear_models_n_params

				svm_models_n_params = [
						(SVC,
						 {'C':C}),

						(NuSVC,
						 {'nu': nu}),

						(LinearSVC,
						 {'penalty_12': penalty_12, 'loss': ['hinge', 'squared_hinge']})
				]

				svm_models_n_params_small = [
						(SVC,
						 {'C':C}),

						(NuSVC,
						 {'nu': nu}),

						(LinearSVC,
						 {'penalty': penalty_12, 'tol': tol})
				]

				neighbor_models_n_params = [

						(KMeans,
						 {'algorithm': ['auto', 'full', 'elkan'],
							'init': ['k-means++', 'random']}),

						(KNeighborsClassifier,
						 {'n_neighbors': n_neighbors, 'algo': neighbor_algo, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
							'weights': ['uniform', 'distance'],
							'p': [1, 2]
							}),

						(NearestCentroid,
						 {'metric': neighbor_metric,
							'shrink_threshold': [1e-3, 1e-2, 0.1, 0.5, 0.9, 2]
							})]

						# not using radius neighbors classifier because it doesn't seem to converge on some of these datasets
						# (RadiusNeighborsClassifier,
						 # {'radius': neighbor_radius, 'algo': neighbor_algo, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
							# 'weights': ['uniform', 'distance'],
							# 'p': [1, 2],
							# 'outlier_label': [-1]
							# })

				gaussianprocess_models_n_params = [
						(GaussianProcessClassifier,
						 {'warm_start': warm_start,
							'kernel': [RBF(), ConstantKernel(), DotProduct(), WhiteKernel()],
							'max_iter_predict': [500],
							'n_restarts_optimizer': [3],
							})
				]

				bayes_models_n_params = [
						(GaussianNB, {})
				]

				nn_models_n_params = [
						(MLPClassifier,
						 { 'hidden_layer_sizes': [(16,), (64,), (100,), (32, 32)],
							 'activation': ['identity', 'logistic', 'tanh', 'relu'],
							 'alpha': alpha, 'learning_rate': learning_rate, 'tol': tol, 'warm_start': warm_start,
							 'batch_size': ['auto', 50],
							 'max_iter': [1000],
							 'early_stopping': [True, False],
							 'epsilon': [1e-8, 1e-5]
							 })
				]

				nn_models_n_params_small = [
						(MLPClassifier,
						 { 'hidden_layer_sizes': [(64,), (32, 64)],
							 'batch_size': ['auto', 50],
							 'activation': ['identity', 'tanh', 'relu'],
							 'max_iter': [500],
							 'early_stopping': [True],
							 'learning_rate': learning_rate_small
							 })
				]

				tree_models_n_params = [

						(RandomForestClassifier,
						 {'criterion': ['gini', 'entropy'],
							'max_features': max_features, 'n_estimators': n_estimators, 'max_depth': max_depth,
							'min_samples_split': min_samples_split, 'min_impurity_split': min_impurity_split, 'warm_start': warm_start, 'min_samples_leaf': min_samples_leaf,
							}),

						(DecisionTreeClassifier,
						 {'criterion': ['gini', 'entropy'],
							'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_impurity_split':min_impurity_split, 'min_samples_leaf': min_samples_leaf
							}),

						(ExtraTreesClassifier,
						 {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
							'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'min_impurity_split': min_impurity_split, 'warm_start': warm_start,
							'criterion': ['gini', 'entropy']})
				]

				tree_models_n_params_small = [

						(RandomForestClassifier,
						 {'max_features_small': max_features_small, 'n_estimators_small': n_estimators_small, 'min_samples_split': min_samples_split, 'max_depth_small': max_depth_small, 'min_samples_leaf': min_samples_leaf
							}),

						(DecisionTreeClassifier,
						 {'max_features_small': max_features_small, 'max_depth_small': max_depth_small, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf
							}),

						(ExtraTreesClassifier,
						 {'n_estimators_small': n_estimators_small, 'max_features_small': max_features_small, 'max_depth_small': max_depth_small,
							'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf})
				]



				def run_all_classifiers(x, y, small = True, normalize_x = True, n_jobs=cpu_count()-1, brain=False, test_size=0.2, n_splits=5, upsample=True, scoring=None, verbose=False, grid_search=True):
						all_params = (linear_models_n_params_small if small else linear_models_n_params) +  (nn_models_n_params_small if small else nn_models_n_params) + (gaussianprocess_models_n_params) + (svm_models_n_params_small if small else svm_models_n_params) + (neighbor_models_n_params) + (tree_models_n_params_small if small else tree_models_n_params)
						# TEST
						# all_params = (linear_models_n_params_small if small else linear_models_n_params)
						return main_loop(all_params, StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=True, n_jobs=n_jobs, verbose=verbose, brain=brain, test_size=test_size, n_splits=n_splits, upsample=upsample, scoring=scoring, grid_search=grid_search)

				class HungaBungaClassifier(ClassifierMixin):
						def __init__(self, brain=False, test_size = 0.2, n_splits = 5, random_state=None, upsample=True, scoring=None, verbose=False, normalize_x = True, n_jobs =cpu_count() - 1, grid_search=True):
								self.model = None
								self.brain = brain
								self.test_size = test_size
								self.n_splits = n_splits
								self.random_state = random_state
								self.upsample = upsample
								self.scoring = None
								self.verbose = verbose
								self.n_jobs = n_jobs
								self.normalize_x = normalize_x
								self.grid_search = grid_search
								super(HungaBungaClassifier, self).__init__()

						def fit(self, x, y):
								self.model = run_all_classifiers(x, y, normalize_x=self.normalize_x, test_size=self.test_size, n_splits=self.n_splits, upsample=self.upsample, scoring=self.scoring, verbose=self.verbose, brain=self.brain, n_jobs=self.n_jobs, grid_search=self.grid_search)[0]
								return self

						def predict(self, x):
								return self.model.predict(x)

				clf = HungaBungaClassifier(brain=False)

		elif mtype in ['regression','r']:
			# regression path

			linear_models_n_params = [
						(LinearRegression, {'normalize': normalize}),

						(Ridge,
						 {'alpha': alpha, 'normalize': normalize, 'tol': tol,
							'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
							}),

						(Lasso,
						 {'alpha': alpha, 'normalize': normalize, 'tol': tol, 'warm_start': warm_start
							}),

						(ElasticNet,
						 {'alpha': alpha, 'normalize': normalize, 'tol': tol,
							'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
							}),

						(Lars,
						 {'normalize': normalize,
							'n_nonzero_coefs': [100, 300, 500, np.inf],
							}),

						(LassoLars,
						 { 'max_iter_inf': max_iter_inf, 'normalize': normalize, 'alpha': alpha
							}),

						(OrthogonalMatchingPursuit,
						 {'n_nonzero_coefs': [100, 300, 500, np.inf, None],
							'tol': tol, 'normalize': normalize
							}),

						(BayesianRidge,
						 {
								 'n_iter': [100, 300, 1000],
								 'tol': tol, 'normalize': normalize,
								 'alpha_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
								 'alpha_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
								 'lambda_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
								 'lambda_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
						 }),

						# WARNING: ARDRegression takes a long time to run
						(ARDRegression,
						 {'n_iter': [100, 300, 1000],
							'tol': tol, 'normalize': normalize,
							'alpha_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
							'alpha_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
							'lambda_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
							'lambda_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
							'threshold_lambda': [1e2, 1e3, 1e4, 1e6]}),

						(SGDRegressor,
						 {'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
							'penalty': penalty_12e, 'n_iter': n_iter, 'epsilon': epsilon, 'eta0': eta0,
							'alpha': [1e-6, 1e-5, 1e-2, 'optimal'],
							'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
							'learning_rate': ['constant', 'optimal', 'invscaling'],
							'power_t': [0.1, 0.25, 0.5]
							}),

						(PassiveAggressiveRegressor,
						 {'C': C, 'epsilon': epsilon, 'n_iter': n_iter, 'warm_start': warm_start,
							'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
							}),

						(RANSACRegressor,
						 {'min_samples': [0.1, 0.5, 0.9, None],
							'max_trials': n_iter,
							'stop_score': [0.8, 0.9, 1],
							'stop_probability': [0.9, 0.95, 0.99, 1],
							'loss': ['absolute_loss', 'squared_loss']
							}),

						(HuberRegressor,
						 { 'epsilon': [1.1, 1.35, 1.5, 2],
							 'max_iter': max_iter, 'alpha': alpha, 'warm_start': warm_start, 'tol': tol
							 }),

						(KernelRidge,
						 {'alpha': alpha, 'degree': degree, 'gamma': gamma, 'coef0': coef0
							})
				]

			linear_models_n_params_small = [
					(LinearRegression, {'normalize': normalize}),

					(Ridge,
					 {'alpha': alpha_small, 'normalize': normalize
						}),

					(Lasso,
					 {'alpha': alpha_small, 'normalize': normalize
						}),

					(ElasticNet,
					 {'alpha': alpha, 'normalize': normalize,
						'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
						}),

					(Lars,
					 {'normalize': normalize,
						'n_nonzero_coefs': [100, 300, 500, np.inf],
						}),

					(LassoLars,
					 {'normalize': normalize, 'max_iter': max_iter_inf, 'alpha': alpha_small
						}),

					(OrthogonalMatchingPursuit,
					 {'n_nonzero_coefs': [100, 300, 500, np.inf, None],
						'normalize': normalize
						}),

					(BayesianRidge,
					 { 'n_iter': [100, 300, 1000],
						 'alpha_1': [1e-6, 1e-3],
						 'alpha_2': [1e-6, 1e-3],
						 'lambda_1': [1e-6, 1e-3],
						 'lambda_2': [1e-6, 1e-3],
						 'normalize': normalize,
						 }),

					# WARNING: ARDRegression takes a long time to run
					(ARDRegression,
					 {'n_iter': [100, 300],
						'normalize': normalize,
						'alpha_1': [1e-6, 1e-3],
						'alpha_2': [1e-6, 1e-3],
						'lambda_1': [1e-6, 1e-3],
						'lambda_2': [1e-6, 1e-3],
						}),

					(SGDRegressor,
					 {'loss': ['squared_loss', 'huber'],
						'penalty': penalty_12e, 'n_iter': n_iter,
						'alpha': [1e-6, 1e-5, 1e-2, 'optimal'],
						'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
						}),

					(PassiveAggressiveRegressor,
					 {'C': C, 'n_iter': n_iter,
						}),

					(RANSACRegressor,
					 {'min_samples': [0.1, 0.5, 0.9, None],
						'max_trials': n_iter,
						'stop_score': [0.8, 1],
						'loss': ['absolute_loss', 'squared_loss']
						}),

					(HuberRegressor,
					 { 'max_iter': max_iter, 'alpha_small': alpha_small,
						 }),

					(KernelRidge,
					 {'alpha': alpha_small, 'degree': degree,
						})
			]

			svm_models_n_params_small = [
					(SVR,
					 {'kernel': kernel, 'degree': degree, 'shrinking': shrinking
						}),

					(NuSVR,
					 {'nu': nu_small, 'kernel': kernel, 'degree': degree, 'shrinking': shrinking,
						}),

					(LinearSVR,
					 {'C': C_small, 'epsilon': epsilon,
						'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
						'intercept_scaling': [0.1, 1, 10]
						})
			]

			svm_models_n_params = [
					(SVR,
					 {'C': C, 'epsilon': epsilon, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol, 'max_iter': max_iter_inf2
						}),

					(NuSVR,
					 {'C': C, 'epsilon': epsilon, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol, 'max_iter': max_iter_inf2
						}),

					(LinearSVR,
					 {'C': C, 'epsilon': epsilon, 'tol': tol, 'max_iter': max_iter,
						'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
						'intercept_scaling': [0.1, 0.5, 1, 5, 10]
						})
			]

			neighbor_models_n_params = [
					(RadiusNeighborsRegressor,
					 {'radius': neighbor_radius, 'algo': neighbor_algo, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
						'weights': ['uniform', 'distance'],
						'p': [1, 2],
						}),

					(KNeighborsRegressor,
					 {'n_neighbors': n_neighbors, 'algo': neighbor_algo, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
						'p': [1, 2],
						'weights': ['uniform', 'distance'],
						})
			]

			gaussianprocess_models_n_params = [
					(GaussianProcessRegressor,
					 {'kernel': [RBF(), ConstantKernel(), DotProduct(), WhiteKernel()],
						'n_restarts_optimizer': [3],
						'alpha': [1e-10, 1e-5],
						'normalize_y': [True, False]
						})
			]

			nn_models_n_params = [
					(MLPRegressor,
					 { 'hidden_layer_sizes': [(16,), (64,), (100,), (32, 64)],
						 'activation': ['identity', 'logistic', 'tanh', 'relu'],
						 'alpha': alpha, 'learning_rate': learning_rate, 'tol': tol, 'warm_start': warm_start,
						 'batch_size': ['auto', 50],
						 'max_iter': [1000],
						 'early_stopping': [True, False],
						 'epsilon': [1e-8, 1e-5]
						 })
			]

			nn_models_n_params_small = [
					(MLPRegressor,
					 { 'hidden_layer_sizes': [(64,), (32, 64)],
						 'activation': ['identity', 'tanh', 'relu'],
						 'max_iter': [500],
						 'early_stopping': [True],
						 'learning_rate': learning_rate_small
						 })
			]

			tree_models_n_params = [

					(DecisionTreeRegressor,
					 {'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'min_impurity_split': min_impurity_split,
						'criterion': ['mse', 'mae']}),

					(ExtraTreesRegressor,
					 {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
						'min_samples_leaf': min_samples_leaf, 'min_impurity_split': min_impurity_split, 'warm_start': warm_start,
						'criterion': ['mse', 'mae']}),

			]

			tree_models_n_params_small = [
					(DecisionTreeRegressor,
					 {'max_features': max_features_small, 'max_depth': max_depth_small, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
						'criterion': ['mse', 'mae']}),

					(ExtraTreesRegressor,
					 {'n_estimators': n_estimators_small, 'max_features': max_features_small, 'max_depth': max_depth_small, 'min_samples_split': min_samples_split,
						'min_samples_leaf': min_samples_leaf,
						'criterion': ['mse', 'mae']})
			]
			def run_all_regressors(x, y, small = True, normalize_x = True, n_jobs=cpu_count()-1, brain=False, test_size=0.2, n_splits=5, upsample=True, scoring=None, verbose=False, grid_search=True):
					all_params = (linear_models_n_params_small if small else linear_models_n_params) + (nn_models_n_params_small if small else nn_models_n_params) + ([] if small else gaussianprocess_models_n_params) + neighbor_models_n_params + (svm_models_n_params_small if small else svm_models_n_params) + (tree_models_n_params_small if small else tree_models_n_params)
					return main_loop(all_params, StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False, n_jobs=n_jobs, brain=brain, test_size=test_size, n_splits=n_splits, upsample=upsample, scoring=scoring, verbose=verbose, grid_search=grid_search)


			class HungaBungaRegressor(RegressorMixin):
					def __init__(self, brain=False, test_size = 0.2, n_splits = 5, random_state=None, upsample=True, scoring=None, verbose=False, normalize_x = True, n_jobs =cpu_count() - 1, grid_search=True):
							self.model = None
							self.brain = brain
							self.test_size = test_size
							self.n_splits = n_splits
							self.random_state = random_state
							self.upsample = upsample
							self.scoring = None
							self.verbose = verbose
							self.n_jobs = n_jobs
							self.normalize_x = normalize_x
							self.grid_search=grid_search
							super(HungaBungaRegressor, self).__init__()

					def fit(self, x, y):
							self.model = run_all_regressors(x, y, normalize_x=self.normalize_x, test_size=self.test_size, n_splits=self.n_splits, upsample=self.upsample, scoring=self.scoring, verbose=self.verbose, brain=self.brain, n_jobs=self.n_jobs, grid_search=self.grid_search)[0]
							return self

					def predict(self, x):
							return self.model.predict(x)

			clf = HungaBungaRegressor(brain=True)

		# write model to .pickle file
		clf.fit(X_train, y_train)

		# now save the model
		f=open(model_name,'wb')
		pickle.dump(clf.model,f)
		f.close()

		# files to put into model folder
		files.append(model_name)
		model_dir=os.getcwd()
		
		return model_name, model_dir, files
