'''
Feature selectors:
- NNI - # https://nni.readthedocs.io/en/latest/FeatureEngineering/GradientFeatureSelector.html
- Scikit-Learn - 
'''

import json, os, sys
import numpy as np 
import helpers.transcribe as ts
from nni.feature_engineering.gradient_selector import FeatureGradientSelector, GBDTSelector
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

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

def feature_select(feature_selector, X_train, y_train):

	# feature_engineering.gradient.selector 
	
	if feature_selector == 'chi':

		'''
		This score can be used to select the n_features features with the 
		highest values for the test chi-squared statistic from X, which must 
		contain only non-negative features such as booleans or frequencies 
		(e.g., term counts in document classification), relative to the classes.

		Recall that the chi-square test measures dependence between stochastic variables, 
		so using this function “weeds out” the features that are the most 
		likely to be independent of class and therefore irrelevant for classification.

		http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
		'''

		X_train, X_test, y_train, y_test = train_test_split(training, labels, test_size=0.20, random_state=42)
		X_train=np.array(X_train)
		X_test=np.array(X_test)
		y_train=np.array(y_train).astype(int)
		y_test=np.array(y_test).astype(int)

		# normalize features so they are non-negative [0,1], or chi squared test will fail
		# it assumes all values are positive 
		min_max_scaler = preprocessing.MinMaxScaler()
		chi_train = min_max_scaler.fit_transform(X_train)
		chi_labels = y_train 

		# Select 50 features with highest chi-squared statistics
		chi2_selector = SelectKBest(chi2, k=50)
		X_kbest = chi2_selector.fit_transform(chi_train, chi_labels)


	elif feature_selector == 'gradient':
		fgs = FeatureGradientSelector(n_features=10)
		fgs.fit(X_train, y_train)
		print(fgs.get_selected_features())

	elif feature_selector == 'gbdt':
		fgs = GBDTSelector()
		fgs.fit(X_train, y_train)
		print(fgs.get_selected_features(10))

	elif feature_selector == 'kbest':

		selector = SelectKBest(f_classif, k=4)
		selector.fit(X_train, y_train)
		scores = -np.log10(selector.pvalues_)
		scores /= scores.max()

	elif feature_selector == 'rfe':
		'''
		Recursive feature elmination works by recursively removing 
		attributes and building a model on attributes that remain. 
		It uses model accuracy to identify which attributes
		(and combinations of attributes) contribute the most to predicting the
		target attribute. You can learn more about the RFE class in
		the scikit-learn documentation.
		'''

		model = LogisticRegression() 
		rfe = RFE(model, 50)
		fit = rfe.fit(X_train, y_train)

		# list out number of features and selected features 
		print("Num Features: %d"% fit.n_features_) 
		print("Selected Features: %s"% fit.support_) 
		print("Feature Ranking: %s"% fit.ranking_)
	

	elif feature_selector == 'lasso':
		# lasso technique 
		'''
		Reconstruction with L1 (Lasso) penalization
		the best value of alpha can be determined using cross validation
		with LassoCV

		http://scikit-learn.org/stable/modules/feature_selection.html#l1-feature-selection
		https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/
		'''
		lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
		model = SelectFromModel(lsvc, prefit=True)
		X_new = model.transform(X_train)
		print(X_new.shape)
