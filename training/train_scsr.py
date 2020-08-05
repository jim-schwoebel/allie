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

Train models using SCSR: https://github.com/jim-schwoebel/voicebook/blob/master/chapter_4_modeling/train_audioregression.py

This is enabled if the default_training_script = ['scsr']
'''
import os
os.system('pip3 install scikit-learn==0.22.2.post1')
os.system('pip3 install xslxwriter==1.2.8')
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from textblob import TextBlob
from operator import itemgetter
import json, pickle, datetime, time, shutil, xlsxwriter
import numpy as np 
from beautifultable import BeautifulTable
import warnings

# ignore a lot of the warnings. 
warnings.filterwarnings("ignore")

# INITIAL FUNCTIONS
#############################################################
def update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores):

	try:
		explained_variances.append(metrics.explained_variance_score(y_test,predictions))
	except:
		explained_variances.append('n/a')
	try:
		mean_absolute_errors.append(metrics.mean_absolute_error(y_test,predictions))
	except:
		mean_absolute_errors.append('n/a')
	try:
		median_absolute_errors.append(metrics.median_absolute_error(y_test,predictions))
	except:
		median_absolute_errors.append('n/a')
	try:
		r2_scores.append(metrics.r2_score(y_test,predictions))
	except:
		r2_scores.append('n/a')

	return explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores

def train_sr(X_train,X_test,y_train,y_test,common_name_model,problemtype,classes,default_features,transform_model,modeldir,settings):

	# metrics 
	modeltypes=list()
	explained_variances=list()
	mean_absolute_errors=list()
	mean_squared_errors=list()
	median_absolute_errors=list()
	r2_scores=list()

	print(modeldir)
	os.chdir(modeldir)

	# make a temp folder to dump files into
	foldername=''
	foldername=common_name_model+'_temp'
	tempdir=os.getcwd()+'/'+foldername 

	try:
		os.mkdir(foldername)
		os.chdir(foldername)
	except:
		shutil.rmtree(foldername)
		os.mkdir(foldername)
		os.chdir(foldername)

	# metrics.explained_variance_score(y_true, y_pred)  Explained variance regression score function
	# metrics.mean_absolute_error(y_true, y_pred)   Mean absolute error regression loss
	# metrics.mean_squared_error(y_true, y_pred[, …])   Mean squared error regression loss
	# metrics.mean_squared_log_error(y_true, y_pred)    Mean squared logarithmic error regression loss
	# metrics.median_absolute_error(y_true, y_pred) Median absolute error regression loss
	# metrics.r2_score(y_true, y_pred[, …]) R^2 (coefficient of determination) regression score function.

	##################################################
	##               linear regression              ##
	##################################################
	'''
	LinearRegression fits a linear model with coefficients w = (w_1, ..., w_p)
	to minimize the residual sum of squares between the observed responses
	in the dataset, and the responses predicted by the linear approximation.

	Example:
	http://scikit-learn.org/stable/modules/linear_model.html
	'''
	try:
		ols = linear_model.LinearRegression()
		ols.fit(X_train, y_train)
		#ols.predict(X_test, y_test)
		predictions = cross_val_predict(ols, X_test, y_test, cv=6)
		f=open('ols.pickle','wb')
		pickle.dump(ols,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('linear regression')
	except:
		print('error - ORDINARY LEAST SQUARES')

	##################################################
	##              Ridge regression                ##
	##################################################
	'''
	Ridge regression addresses some of the problems of
	Ordinary Least Squares by imposing a penalty on the
	size of coefficients.

	The ridge coefficients minimize a penalized residual sum of squares.

	Example:
	http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge

	'''
	try:
		ridge = linear_model.Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True)
		ridge.fit(X_train, y_train)
		predictions = cross_val_predict(ridge, X_test, y_test, cv=6)
		f=open('ridge.pickle','wb')
		pickle.dump(ridge,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('ridge regression')
	except:
		print('error - RIDGE REGRESSION')

	##################################################
	##                    LASSO                     ##
	##################################################
	'''
	The Lasso is a linear model that estimates sparse coefficients.
	It is useful in some contexts due to its tendency to prefer solutions
	with fewer parameter values, effectively reducing the number of
	variables upon which the given solution is dependent.

	For this reason, the Lasso and its variants are fundamental
	to the field of compressed sensing. Under certain conditions,
	it can recover the exact set of non-zero weights
	(see Compressive sensing: tomography reconstruction with L1 prior (Lasso)).

	Example:
	http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py

	'''
	try:
		lasso = linear_model.Lasso(alpha = 0.1)
		lasso.fit(X_train, y_train)
		predictions = cross_val_predict(lasso, X_test, y_test, cv=6)
		f=open('lasso.pickle','wb')
		pickle.dump(lasso,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('LASSO')
	except:
		print('error - LASSO')

	##################################################
	##              Multi-task LASSO                ##
	##################################################
	'''
	The MultiTaskLasso is a linear model that estimates
	sparse coefficients for multiple regression problems
	jointly: y is a 2D array, of shape (n_samples, n_tasks).
	The constraint is that the selected features are the same
	for all the regression problems, also called tasks.

	Example:
	http://scikit-learn.org/stable/auto_examples/linear_model/plot_multi_task_lasso_support.html#sphx-glr-auto-examples-linear-model-plot-multi-task-lasso-support-py

	'''
	# # ONLY WORKS ON y_train that is multidimensional (one hot encoded)
	# # Generate some 2D coefficients with sine waves with random frequency and phase
	# mlasso = linear_model.MultiTaskLasso(alpha=0.1)
	# mlasso.fit(X_train, y_train)
	# predictions = cross_val_predict(mlasso, X_test, y_test, cv=6)
	# accuracy = metrics.r2_score(y_test, predictions)

	##################################################
	##                  Elastic net                 ##
	##################################################
	'''
	ElasticNet is a linear regression model trained with L1 and L2 prior as regularizer.
	This combination allows for learning a sparse model where few of the weights are non-zero
	like Lasso, while still maintaining the regularization properties of Ridge.

	We control the convex combination of L1 and L2 using the l1_ratio parameter.

	Example:
	http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py

	'''
	# need training data 
	try:
		enet = linear_model.ElasticNet()
		enet.fit(X_train, y_train)
		predictions = cross_val_predict(enet, X_test, y_test, cv=6)
		f=open('enet.pickle','wb')
		pickle.dump(enet,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(ytest, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('elastic net')
	except:
		print('error - ELASTIC NET')

	##################################################
	##            Multi-task elastic net            ##
	##################################################
	'''
	The MultiTaskElasticNet is an elastic-net model that estimates sparse coefficients
	for multiple regression problems jointly: Y is a 2D array, of shape (n_samples, n_tasks).

	The constraint is that the selected features are the same for all the regression problems,
	also called tasks.

	Example:
	http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html
	'''
	# # # ONLY WORKS ON y_train that is multidimensional (one hot encoded)
	# clf = linear_model.MultiTaskElasticNet()
	# clf.fit(X_train, y_train)
	# #print(clf.coef_)
	# #print(clf.intercept_)

	##################################################
	##          Least angle regression (LARS)       ##
	##################################################
	'''
	The advantages of LARS are:

	-> It is numerically efficient in contexts where p >> n (i.e., when the number of dimensions is significantly greater than the number of points)
	-> It is computationally just as fast as forward selection and has the same order of complexity as an ordinary least squares.
	-> It produces a full piecewise linear solution path, which is useful in cross-validation or similar attempts to tune the model.
	-> If two variables are almost equally correlated with the response, then their coefficients should increase at approximately the same rate. The algorithm thus behaves as intuition would expect, and also is more stable.
	-> It is easily modified to produce solutions for other estimators, like the Lasso.

	The disadvantages of the LARS method include:

	-> Because LARS is based upon an iterative refitting of the residuals,
	-> it would appear to be especially sensitive to the effects of noise.

	Example:
	http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html
	'''
	try:
		lars = linear_model.Lars(n_nonzero_coefs=1)
		lars.fit(X_train, y_train)
		predictions = cross_val_predict(lars, X_test, y_test, cv=6)
		f=open('lars.pickle','wb')
		pickle.dump(lars,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('Least angle regression (LARS)')
	except:
		print('error - LARS')

	##################################################
	##                 LARS LASSO                   ##
	##################################################
	'''
	LassoLars is a lasso model implemented using the LARS algorithm,
	and unlike the implementation based on coordinate_descent,
	this yields the exact solution, which is piecewise linear
	as a function of the norm of its coefficients.

	Example:
	http://scikit-learn.org/stable/modules/linear_model.html#passive-aggressive-algorithms

	'''
	try:
		lars_lasso = linear_model.LassoLars()
		lars_lasso.fit(X_train, y_train)
		predictions = cross_val_predict(lars_lasso, X_test, y_test, cv=6)
		f=open('lars_lasso.pickle','wb')
		pickle.dump(lars_lasso,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('LARS lasso')
	except:
		print('error - LARS LASSO')

	##################################################
	##      Orthogonal Matching Pursuit (OMP)       ##
	##################################################
	'''
	OrthogonalMatchingPursuit and orthogonal_mp implements the OMP
	algorithm for approximating the fit of a linear model with
	constraints imposed on the number of non-zero coefficients (ie. the L 0 pseudo-norm).

	Example:
	http://scikit-learn.org/stable/auto_examples/linear_model/plot_omp.html#sphx-glr-auto-examples-linear-model-plot-omp-py
	'''
	try:
		omp = linear_model.OrthogonalMatchingPursuit()
		omp.fit(X_train, y_train)
		predictions = cross_val_predict(omp, X_test, y_test, cv=6)
		f=open('omp.pickle','wb')
		pickle.dump(omp,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('orthogonal matching pursuit (OMP)')
	except:
		print('error - ORTHOGONAL MATCHING PURSUIT (OMP)')

	##################################################
	##          Bayesian ridge regression           ##
	##################################################
	'''
	The advantages of Bayesian Regression are:

	-> It adapts to the data at hand.
	-> It can be used to include regularization parameters in the estimation procedure.

	The disadvantages of Bayesian regression include:

	-> Inference of the model can be time consuming.

	Example:
	http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
	'''
	# MULTI-DIMENSIONAL 
	# clf = BayesianRidge()
	# clf.fit(X_train, y_train)
	# predictions = cross_val_predict(clf, X_test, y_test, cv=6)
	# accuracy = metrics.r2_score(y_test, predictions)

	##################################################
	##      Automatic relevance determination       ## 
	##################################################
	'''
	ARDRegression is very similar to Bayesian Ridge Regression,
	but can lead to sparser weights w [1] [2]. ARDRegression poses
	a different prior over w, by dropping the assumption of
	the Gaussian being spherical.

	Example:
	http://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html#sphx-glr-auto-examples-linear-model-plot-ard-py
	'''
	# MULTI-DIMENSIONAL
	# clf = ARDRegression(compute_score=True)
	# clf.fit(X_train, y_train)
	# predictions = cross_val_predict(clf, X_test, y_test, cv=6)
	# accuracy = metrics.r2_score(y_test, predictions)

	##################################################
	##              Logistic regression             ##
	##################################################
	'''
	Logistic regression, despite its name, is a linear model
	for classification rather than regression. Logistic regression
	is also known in the literature as logit regression,
	maximum-entropy classification (MaxEnt) or the log-linear classifier.

	In this model, the probabilities describing the possible outcomes
	of a single trial are modeled using a logistic function.

	Example:
	http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html#sphx-glr-auto-examples-linear-model-plot-logistic-l1-l2-sparsity-py
	'''
	try:
		lr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
		lr.fit(X_train, y_train)
		predictions = cross_val_predict(lr, X_test, y_test, cv=6)
		f=open('lr.pickle','wb')
		pickle.dump(lr,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('logistic regression')
	except:
		print('error - LOGISTIC REGRESSION')

	##################################################
	##      Stochastic gradient descent (SGD)       ##
	##################################################
	'''
	Stochastic gradient descent is a simple yet very efficient
	approach to fit linear models. It is particularly useful
	when the number of samples (and the number of features) is very large.
	The partial_fit method allows only/out-of-core learning.

	The classes SGDClassifier and SGDRegressor provide functionality
	to fit linear models for classification and regression using
	different (convex) loss functions and different penalties.
	E.g., with loss="log", SGDClassifier fits a logistic regression model,
	while with loss="hinge" it fits a linear support vector machine (SVM).

	Example:
	http://scikit-learn.org/stable/modules/sgd.html#sgd
	'''
	try:
		# note you have to scale the data, as SGD algorithms are sensitive to 
		# feature scaling 
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train_2 = scaler.transform(X_train)
		X_test_2 = scaler.transform(X_test) 
		sgd = linear_model.SGDRegressor()
		sgd.fit(X_train_2, y_train)
		predictions = cross_val_predict(sgd, X_test_2, y_test, cv=6)
		f=open('sgd.pickle','wb')
		pickle.dump(sgd,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('stochastic gradient descent (SGD)')
	except: 
		print('error - STOCHASTIC GRADIENT DESCENT')

	##################################################
	##          Perceptron algorithms               ## 
	##################################################
	'''
	Multi-layer Perceptron is sensitive to feature scaling,
	so it is highly recommended to scale your data.
	For example, scale each attribute on the input vector X to [0, 1] or [-1, +1],
	or standardize it to have mean 0 and variance 1.

	Note that you must apply the same scaling to the test
	set for meaningful results. You can use StandardScaler for standardization.

	change the solver to 'lbfgs'. The default'adam' is a SGD-like method,
	hich is effective for large & messy data but pretty useless for this kind of smooth & small data.

	Example:
	http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor
	'''
	try:
		nn = MLPRegressor(solver='lbfgs')
		nn.fit(X_train, y_train)
		predictions = cross_val_predict(nn, X_test, y_test, cv=6)
		f=open('nn.pickle','wb')
		pickle.dump(nn,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('perceptron')
	except:
		print('error - MLP REGRESSOR')

	##################################################
	##          Passive-agressive algorithms        ##
	##################################################
	'''
	The passive-aggressive algorithms are a family of algorithms
	for large-scale learning. They are similar to the Perceptron
	in that they do not require a learning rate. However,
	contrary to the Perceptron, they include a regularization parameter C.

	Example:
	http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf
	'''
	try:
		pa_regr = linear_model.PassiveAggressiveRegressor(random_state=0)
		pa_regr.fit(X_train, y_train)
		predictions = cross_val_predict(pa_regr, X_test, y_test, cv=6)
		f=open('pa_regr.pickle','wb')
		pickle.dump(pa_regr,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('passive-agressive algorithm')
	except:
		print('error - PASSIVE-AGGRESSIVE')

	##################################################
	##                   RANSAC                     ## 
	##################################################
	'''
	When in doubt, use RANSAC

	RANSAC (RANdom SAmple Consensus) fits a model from random subsets of
	inliers from the complete data set.

	RANSAC is a non-deterministic algorithm producing only a reasonable
	result with a certain probability, which is dependent on the number
	of iterations (see max_trials parameter). It is typically used for
	linear and non-linear regression problems and is especially popular
	in the fields of photogrammetric computer vision.

	Example:
	http://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html#sphx-glr-auto-examples-linear-model-plot-ransac-py
	'''
	try:
		ransac = linear_model.RANSACRegressor()
		ransac.fit(X_train, y_train)
		predictions = cross_val_predict(ransac, X_test, y_test, cv=6)
		f=open('ransac.pickle','wb')
		pickle.dump(ransac,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('RANSAC')
	except:
		print('error - RANSAC')


	##################################################
	##              Theil-SEN                       ##
	##################################################
	'''
	The TheilSenRegressor estimator uses a generalization of the median
	in multiple dimensions. It is thus robust to multivariate outliers.

	Note however that the robustness of the estimator decreases quickly
	with the dimensionality of the problem. It looses its robustness
	properties and becomes no better than an ordinary least squares
	in high dimension.

	Note takes a bit longer to train.

	Example:
	http://scikit-learn.org/stable/auto_examples/linear_model/plot_theilsen.html#sphx-glr-auto-examples-linear-model-plot-theilsen-py

	'''
	try:
		theilsen=linear_model.TheilSenRegressor(random_state=42)
		theilsen.fit(X_train, y_train)
		predictions = cross_val_predict(theilsen, X_test, y_test, cv=6)
		f=open('theilsen.pickle','wb')
		pickle.dump(theilsen,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('Theil-Sen')
	except:
		print('error - THEILSEN')

	##################################################
	##              Huber Regression                ##
	##################################################
	'''
	The HuberRegressor is different to Ridge because it applies a linear loss
	to samples that are classified as outliers. A sample is classified as an
	inlier if the absolute error of that sample is lesser than a certain threshold.

	It differs from TheilSenRegressor and RANSACRegressor because it does not
	ignore the effect of the outliers but gives a lesser weight to them.

	Example:
	http://scikit-learn.org/stable/auto_examples/linear_model/plot_huber_vs_ridge.html#sphx-glr-auto-examples-linear-model-plot-huber-vs-ridge-py
	'''
	try:
		huber = linear_model.HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100)
		huber.fit(X_train, y_train)
		predictions = cross_val_predict(huber, X_test, y_test, cv=6)
		f=open('huber.pickle','wb')
		pickle.dump(huber,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('huber regression')
	except:
		print('error - HUBER')

	##################################################
	##              Polynomial Regression           ##
	##################################################
	'''
	One common pattern within machine learning is to use linear models trained on
	nonlinear functions of the data. This approach maintains the generally fast
	performance of linear methods, while allowing them to fit a much wider range of data.

	Example:
	http://scikit-learn.org/stable/modules/linear_model.html#passive-aggressive-algorithms

	'''
	try:
		poly_lr = Pipeline([
							('poly', PolynomialFeatures(degree=5, include_bias=False)),
							('linreg', LinearRegression(normalize=True))
							])


		poly_lr.fit(X_train, y_train)
		predictions = cross_val_predict(poly_lr, X_test, y_test, cv=6)
		accuracy = metrics.r2_score(y_test, predictions)
		f=open('poly_lr.pickle','wb')
		pickle.dump(poly_lr,f)
		f.close()
		# get stats 
		explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores = update_list(y_test, predictions, explained_variances, mean_absolute_errors, mean_squared_errors, median_absolute_errors, r2_scores)
		modeltypes.append('polynomial (linear regression)')
	except:
		print('error - POLYNOMIAL')

	##################################################
	##              Write session to .JSON          ##
	##################################################

	os.chdir(modeldir)

	print('\n\n')
	print('RESULTS: \n')

	# print table in terminal 
	table = BeautifulTable()
	table.column_headers = ["model type", "R^2 score", "Mean Absolute Errors"]
	print(len(modeltypes))
	print(len(r2_scores))
	print(len(mean_absolute_errors))

	for i in range(len(modeltypes)):
		table.append_row([modeltypes[i], str(r2_scores[i]), str(mean_absolute_errors[i])])

	print(table)

	filename=common_name_model+'.xlsx'
	workbook  = xlsxwriter.Workbook(filename)
	worksheet = workbook.add_worksheet()

	worksheet.write('A1', 'Model type')
	worksheet.write('B1', 'R^2 score')
	worksheet.write('C1', 'Explained Variances')
	worksheet.write('D1', 'Mean Absolute Errors')
	worksheet.write('E1', 'Mean Squared Log Errors')
	worksheet.write('F1', 'Median Absolute Errors')
	#worksheet.write('G1', 'Mean Squared Errors')

	# print the best model in terms of mean abolute error 
	varnames=['ols.pickle', 'ridge.pickle', 'lasso.pickle', 'enet.pickle', 'lars.pickle', 
			  'lars_lasso.pickle','omp.pickle', 'lr.pickle','sgd.pickle', 'nn.pickle','pa_regr.pickle',
			  'ransac.pickle', 'theilsen.pickle', 'huber.pickle', 'poly_lr.pickle']

	# make sure all numbers, make mae 10 (a large number, to eliminate it from the list of options)
	mae=mean_absolute_errors
	for i in range(len(mae)):
		if mae[i] == 'n/a':
			mae[i]=10
		else:
			mae[i]=float(mae[i])

	# get minimim index and now delete temp folder, put master file in models directory 
	minval=np.amin(mae)
	ind=mae.index(minval)
	print('%s has the lowest mean absolute error (%s)'%(modeltypes[ind], str(minval)))
	# rename file 
	os.chdir(tempdir)
	newname= common_name_model+'.pickle'
	print('saving file to disk (%s)...'%(newname))
	os.rename(varnames[ind], newname)
	# move to models directory
	shutil.copy(os.getcwd()+'/'+newname, modeldir+'/'+newname)
	# now delete temp folder 
	os.chdir(modeldir)
	shutil.rmtree(foldername)

	# output spreadsheet of results and open up for analyis
	for i in range(len(modeltypes)):
		try:
			worksheet.write('A'+str(i+2), str(modeltypes[i]))
			worksheet.write('B'+str(i+2), str(r2_scores[i]))
			worksheet.write('C'+str(i+2), str(explained_variances[i]))
			worksheet.write('D'+str(i+2), str(mean_absolute_errors[i]))
			worksheet.write('F'+str(i+2), str(median_absolute_errors[i]))
			#worksheet.write('G'+str(i+2), str(mean_squared_errors[i]))
			
		except:
			pass

	workbook.close()

	files=list()
	files.append(common_name_model+'.xlsx')
	files.append(common_name_model+'.pickle')

	model_name=common_name_model+'.pickle'
	model_dir=os.getcwd()

	return model_name, model_dir, files

def train_sc(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_features,transform_model,settings,min_num):

	# create common_name
	selectedfeature=str(default_features) + ' (%s)'%(problemtype)
	modelname=common_name_model
	training_data='train labels'+'\n\n'+str(y_train)+'\n\n'+'test labels'+'\n\n'+str(y_test)+'\n\n'
	filename=modelname
	start=time.time()
	
	c1=0
	c5=0

	try:
		#decision tree
		classifier2 = DecisionTreeClassifier(random_state=0)
		classifier2.fit(X_train,y_train)
		# cross val score taken from documentation (95% interval) - https://scikit-learn.org/stable/modules/cross_validation.html
		scores = cross_val_score(classifier2, X_test, y_test,cv=5)
		print('Decision tree accuracy (+/-) %s'%(str(scores.std()*2)))
		c2=scores.mean()
		c2s=scores.std()*2 
		print(c2)
	except:
		c2=0
		c2s=0

	try:
		classifier3 = GaussianNB()
		classifier3.fit(X_train,y_train)
		scores = cross_val_score(classifier3, X_test, y_test,cv=5)
		print('Gaussian NB accuracy (+/-) %s'%(str(scores.std()*2)))
		c3=scores.mean()
		c3s=scores.std()*2 
		print(c3)
	except:
		c3=0
		c3s=0

	try:
		#svc 
		classifier4 = SVC()
		classifier4.fit(X_train,y_train)
		scores=cross_val_score(classifier4, X_test, y_test,cv=5)
		print('SKlearn classifier accuracy (+/-) %s'%(str(scores.std()*2)))
		c4=scores.mean()
		c4s=scores.std()*2 
		print(c4)
	except:
		c4=0
		c4s=0

	try:
		#adaboost
		classifier6 = AdaBoostClassifier(n_estimators=100)
		classifier6.fit(X_train,y_train)
		scores = cross_val_score(classifier6, X_test, y_test,cv=5)
		print('Adaboost classifier accuracy (+/-) %s'%(str(scores.std()*2)))
		c6=scores.mean()
		c6s=scores.std()*2 
		print(c6)
	except:
		c6=0
		c6s=0

	try:
		#gradient boosting 
		classifier7=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
		classifier7.fit(X_train,y_train)
		scores = cross_val_score(classifier7, X_test, y_test,cv=5)
		print('Gradient boosting accuracy (+/-) %s'%(str(scores.std()*2)))
		c7=scores.mean()
		c7s=scores.std()*2 
		print(c7)
	except:
		c7=0
		c7s=0

	try:
		#logistic regression
		classifier8=LogisticRegression(random_state=1)
		classifier8.fit(X_train,y_train)
		scores = cross_val_score(classifier8, X_test, y_test,cv=5)
		print('Logistic regression accuracy (+/-) %s'%(str(scores.std()*2)))
		c8=scores.mean()
		c8s=scores.std()*2 
		print(c8)
	except:
		c8=0
		c8s=0

	try:
		#voting 
		classifier9=VotingClassifier(estimators=[('gradboost', classifier7), ('logit', classifier8), ('adaboost', classifier6)], voting='hard')
		classifier9.fit(X_train,y_train)
		scores = cross_val_score(classifier9, X_test, y_test,cv=5)
		print('Hard voting accuracy (+/-) %s'%(str(scores.std()*2)))
		c9=scores.mean()
		c9s=scores.std()*2 
		print(c9)
	except:
		c9=0
		c9s=0

	try:
		#knn
		classifier10=KNeighborsClassifier(n_neighbors=7)
		classifier10.fit(X_train,y_train)
		scores = cross_val_score(classifier10, X_test, y_test,cv=5)
		print('K Nearest Neighbors accuracy (+/-) %s'%(str(scores.std()*2)))
		c10=scores.mean()
		c10s=scores.std()*2 
		print(c10)
	except:
		c10=0
		c10s=0

	try:
		#randomforest
		classifier11=RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
		classifier11.fit(X_train,y_train)
		scores = cross_val_score(classifier11, X_test, y_test,cv=5)
		print('Random forest accuracy (+/-) %s'%(str(scores.std()*2)))
		c11=scores.mean()
		c11s=scores.std()*2 
		print(c11)
	except:
		c11=0
		c11s=0

	try:
##        #svm
		classifier12 = svm.SVC(kernel='linear', C = 1.0)
		classifier12.fit(X_train,y_train)
		scores = cross_val_score(classifier12, X_test, y_test,cv=5)
		print('svm accuracy (+/-) %s'%(str(scores.std()*2)))
		c12=scores.mean()
		c12s=scores.std()*2 
		print(c12)
	except:
		c12=0
		c12s=0

	#IF IMBALANCED, USE http://scikit-learn.org/dev/modules/generated/sklearn.naive_bayes.ComplementNB.html

	maxacc=max([c2,c3,c4,c6,c7,c8,c9,c10,c11,c12])
	X_train=list(X_train)
	X_test=list(X_test)
	y_train=list(y_train)
	y_test=list(y_test)
	# if maxacc==c1:
	#     print('most accurate classifier is Naive Bayes'+'with %s'%(selectedfeature))
	#     classifiername='naive-bayes'
	#     classifier=classifier1
	#     #show most important features
	#     classifier1.show_most_informative_features(5)
	if maxacc==c2:
		print('most accurate classifier is Decision Tree'+'with %s'%(selectedfeature))
		classifiername='decision-tree'
		classifier2 = DecisionTreeClassifier(random_state=0)
		classifier2.fit(X_train+X_test,y_train+y_test)
		classifier=classifier2
	elif maxacc==c3:
		print('most accurate classifier is Gaussian NB'+'with %s'%(selectedfeature))
		classifiername='gaussian-nb'
		classifier3 = GaussianNB()
		classifier3.fit(X_train+X_test,y_train+y_test)
		classifier=classifier3
	elif maxacc==c4:
		print('most accurate classifier is SK Learn'+'with %s'%(selectedfeature))
		classifiername='sk'
		classifier4 = SVC()
		classifier4.fit(X_train+X_test,y_train+y_test)
		classifier=classifier4
	elif maxacc==c5:
		print('most accurate classifier is Maximum Entropy Classifier'+'with %s'%(selectedfeature))
		classifiername='max-entropy'
		classifier=classifier5
	#can stop here (c6-c10)
	elif maxacc==c6:
		print('most accuracate classifier is Adaboost classifier'+'with %s'%(selectedfeature))
		classifiername='adaboost'
		classifier6 = AdaBoostClassifier(n_estimators=100)
		classifier6.fit(X_train+X_test,y_train+y_test)
		classifier=classifier6
	elif maxacc==c7:
		print('most accurate classifier is Gradient Boosting '+'with %s'%(selectedfeature))
		classifiername='graidentboost'
		classifier7=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
		classifier7.fit(X_train+X_test,y_train+y_test)
		classifier=classifier7
	elif maxacc==c8:
		print('most accurate classifier is Logistic Regression '+'with %s'%(selectedfeature))
		classifiername='logistic_regression'
		classifier8=LogisticRegression(random_state=1)
		classifier8.fit(X_train+X_test,y_train+y_test)
		classifier=classifier8
	elif maxacc==c9:
		print('most accurate classifier is Hard Voting '+'with %s'%(selectedfeature))
		classifiername='hardvoting'
		classifier7=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
		classifier7.fit(X_train+X_test,y_train+y_test)
		classifier8=LogisticRegression(random_state=1)
		classifier8.fit(X_train+X_test,y_train+y_test)
		classifier6 = AdaBoostClassifier(n_estimators=100)
		classifier6.fit(X_train+X_test,y_train+y_test)
		classifier9=VotingClassifier(estimators=[('gradboost', classifier7), ('logit', classifier8), ('adaboost', classifier6)], voting='hard')
		classifier9.fit(X_train+X_test,y_train+y_test)
		classifier=classifier9
	elif maxacc==c10:
		print('most accurate classifier is K nearest neighbors '+'with %s'%(selectedfeature))
		classifiername='knn'
		classifier10=KNeighborsClassifier(n_neighbors=7)
		classifier10.fit(X_train+X_test,y_train+y_test)
		classifier=classifier10
	elif maxacc==c11:
		print('most accurate classifier is Random forest '+'with %s'%(selectedfeature))
		classifiername='randomforest'
		classifier11=RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
		classifier11.fit(X_train+X_test,y_train+y_test)
		classifier=classifier11
	elif maxacc==c12:
		print('most accurate classifier is SVM '+' with %s'%(selectedfeature))
		classifiername='svm'
		classifier12 = svm.SVC(kernel='linear', C = 1.0)
		classifier12.fit(X_train+X_test,y_train+y_test)
		classifier=classifier12

	modeltypes=['decision-tree','gaussian-nb','sk','adaboost','gradient boosting','logistic regression','hard voting','knn','random forest','svm']
	accuracym=[c2,c3,c4,c6,c7,c8,c9,c10,c11,c12]
	accuracys=[c2s,c3s,c4s,c6s,c7s,c8s,c9s,c10s,c11s,c12s]
	model_accuracy=list()
	for i in range(len(modeltypes)):
		model_accuracy.append([modeltypes[i],accuracym[i],accuracys[i]])

	model_accuracy.sort(key=itemgetter(1))
	endlen=len(model_accuracy)

	print('saving classifier to disk')
	f=open(modelname+'.pickle','wb')
	pickle.dump(classifier,f)
	f.close()

	end=time.time()

	execution=end-start
	
	print('summarizing session...')

	accstring=''
	
	for i in range(len(model_accuracy)):
		accstring=accstring+'%s: %s (+/- %s)\n'%(str(model_accuracy[i][0]),str(model_accuracy[i][1]),str(model_accuracy[i][2]))

	training=len(X_train)
	testing=len(y_train)
	
	summary='SUMMARY OF MODEL SELECTION \n\n'+'WINNING MODEL: \n\n'+'%s: %s (+/- %s) \n\n'%(str(model_accuracy[len(model_accuracy)-1][0]),str(model_accuracy[len(model_accuracy)-1][1]),str(model_accuracy[len(model_accuracy)-1][2]))+'MODEL FILE NAME: \n\n %s.pickle'%(filename)+'\n\n'+'DATE CREATED: \n\n %s'%(datetime.datetime.now())+'\n\n'+'EXECUTION TIME: \n\n %s\n\n'%(str(execution))+'GROUPS: \n\n'+str(classes)+'\n'+'('+str(min_num)+' in each class, '+str(len(y_test))+'% used for testing)'+'\n\n'+'TRAINING SUMMARY:'+'\n\n'+str(y_train)+'\n\n'+'FEATURES: \n\n %s'%(selectedfeature)+'\n\n'+'MODELS, ACCURACIES, AND STANDARD DEVIATIONS: \n\n'+accstring+'\n\n'+'(C) 2019, NeuroLex Laboratories'

	# write to .JSON and move to proper directory...
	g=open(modelname+'.txt','w')
	g.write(summary)
	g.close()

	files=list()
	files.append(common_name_model+'.txt')
	files.append(common_name_model+'.pickle')

	model_name=common_name_model+'.pickle'
	model_dir=os.getcwd()
	
	return model_name, model_dir, files
