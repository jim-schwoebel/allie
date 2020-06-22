'''
Feature selectors:
- NNI - # https://nni.readthedocs.io/en/latest/FeatureEngineering/GradientFeatureSelector.html
- Scikit-Learn - 
'''

import json, os, sys
import numpy as np 
# from nni.feature_engineering.gradient_selector import FeatureGradientSelector, GBDTSelector
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


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

def feature_select(feature_selector, X_train, y_train, feature_number):

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

		# Select 50 features with highest chi-squared statistics
		model = SelectKBest(chi2, feature_number)

	####################################################################################
	# NNI-based feature selectors cn be chosen into the future
	# elif feature_selector == 'gradient':
		# model = FeatureGradientSelector(n_features=feature_number)

	# elif feature_selector == 'gbdt':
		# model = GBDTSelector(n_features=feature_number)
	####################################################################################

	elif feature_selector == 'kbest':
		model = SelectKBest(f_classif, k=feature_number)

	elif feature_selector == 'rfe':
		'''
		Recursive feature elmination works by recursively removing 
		attributes and building a model on attributes that remain. 
		It uses model accuracy to identify which attributes
		(and combinations of attributes) contribute the most to predicting the
		target attribute. You can learn more about the RFE class in
		the scikit-learn documentation.
		'''
		estimator = SVR(kernel="linear")
		model = RFE(estimator, n_features_to_select=feature_number, step=1)

	elif feature_selector == 'lasso':
		# lasso technique 
		'''
		Reconstruction with L1 (Lasso) penalization
		the best value of alpha can be determined using cross validation
		with LassoCV

		http://scikit-learn.org/stable/modules/feature_selection.html#l1-feature-selection
		https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/
		'''
		lsvc = LinearSVC(C=0.01, penalty="l1", dual=False)
		model = SelectFromModel(lsvc)

	elif feature_selector == 'variance':
		model = VarianceThreshold(threshold=(.8 * (1 - .8)))
		
	return model
