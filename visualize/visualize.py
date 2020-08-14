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
                                                                             
| | | (_)               | (_)          / _ \ | ___ \_   _|
| | | |_ ___ _   _  __ _| |_ _______  / /_\ \| |_/ / | |  
| | | | / __| | | |/ _` | | |_  / _ \ |  _  ||  __/  | |  
\ \_/ / \__ \ |_| | (_| | | |/ /  __/ | | | || |    _| |_ 
 \___/|_|___/\__,_|\__,_|_|_/___\___| \_| |_/\_|    \___/ 
                                                          
Takes in a folder or set of folders of featurized files and outputs
visualizations to look deeper at the data.

This is often useful as a precursor before building machine learning 
models to uncover relationships in the data.

Note that this automatically happens as part of the modeling process
if visualize==True in settings.

This is also restricted to classification problems for version 1.0 of Allie.

Usage: python3 visualize.py [problemtype] [folder A] [folder B] ... [folder N]

Example: python3 visualize.py audio males females 
'''
import os, sys, json, time, shutil
os.system('pip3 install yellowbrick==1.1 scikit-plot==0.3.7 umap==0.1.1 umap-learn==0.4.4')
from tqdm import tqdm
from yellowbrick.features import Rank1D, Rank2D, Manifold, FeatureImportances
from yellowbrick.features.pca import PCADecomposition
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from yellowbrick.text import UMAPVisualizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import precision_recall_curve, discrimination_threshold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from yellowbrick.regressor import residuals_plot
from yellowbrick.regressor import prediction_error
from sklearn.cluster import KMeans
from yellowbrick.cluster import silhouette_visualizer
from sklearn.cluster import MiniBatchKMeans
from yellowbrick.cluster import intercluster_distance
from sklearn.metrics import auc, roc_curve
from yellowbrick.classifier.rocauc import roc_auc
from yellowbrick.regressor import cooks_distance
from yellowbrick.features import RadViz
from yellowbrick.target.feature_correlation import feature_correlation
import seaborn as sns
import umap
from sklearn.model_selection import train_test_split

# feature selection
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# other things in scikitlearn
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve  
from sklearn.cluster import KMeans
from sklearn import metrics
from itertools import cycle

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

def get_classes():
	count=2
	classes=list()
	while True:
		try:
			class_=sys.argv[count]
			classes.append(class_)
			print(classes)
			count=count+1
		except:
			break

	# added this in for the CLI
	if classes == []:
		classnum=input('how many classes do you want to model? (e.g. 2)\n')
		classnum = int(classnum)
		for i in range(classnum):
			classes.append(input('what is class #%s\n'%(str(i+1))))

	return classes

def get_features(classes, problem_type, default_features, balance_data):

	features=list()
	feature_labels=list()
	class_labels=list()
	curdir=os.getcwd()
	lengths=list()
	minlength=0

	if balance_data == True:
		for i in range(len(classes)):
			os.chdir(curdir+'/'+classes[i])
			listdir=os.listdir()
			jsonfiles=list()
			for j in range(len(listdir)):
				if listdir[j].endswith('.json'):
					jsonfiles.append(listdir[j])

			lengths.append(len(jsonfiles))
		minlength=np.amin(lengths)

		print('minimum length is...')
		print(minlength)
		time.sleep(2)

	for i in range(len(classes)):

		print('----------LOADING %s----------'%(classes[i].upper()))
		os.chdir(curdir+'/'+classes[i])
		listdir=os.listdir()
		jsonfiles=list()

		for j in range(len(listdir)):
			if listdir[j].endswith('.json'):
				jsonfiles.append(listdir[j])

		g=json.load(open(jsonfiles[0]))
		feature_list=list(g['features'][problem_type])

		for j in tqdm(range(len(jsonfiles))):
			if balance_data==True:
				if class_labels.count(classes[i]) > minlength:
					break 
				else:
					try:
						g=json.load(open(jsonfiles[j]))
						feature_=list()
						label_=list()
						
						for k in range(len(feature_list)):
							if feature_list[k] in default_features:
								feature_=feature_+g['features'][problem_type][feature_list[k]]['features']
								label_=label_+g['features'][problem_type][feature_list[k]]['labels']

						# quick quality check to only add to list if the feature_labels match in length the features_
						if len(feature_) == len(label_):
							features.append(feature_)
							feature_labels.append(label_)
							class_labels.append(classes[i])
					except:
						print('error loading feature embedding: %s'%(feature_list[k].upper()))

			else:
				try:
					g=json.load(open(jsonfiles[j]))
					feature_=list()
					label_=list()
				
					for k in range(len(feature_list)):
						if feature_list[k] in default_features:
							feature_=feature_+g['features'][problem_type][feature_list[k]]['features']
							label_=label_+g['features'][problem_type][feature_list[k]]['labels']

					# quick quality check to only add to list if the feature_labels match in length the features_
					if len(feature_) == len(label_):
						features.append(feature_)
						feature_labels.append(label_)
						class_labels.append(classes[i])
				except:
					print('error loading feature embedding: %s'%(feature_list[k].upper()))


	return features, feature_labels, class_labels 

def plot_roc_curve(y_test, probs, clf_names):  
	cycol = cycle('bgrcmyk')

	for i in range(len(probs)):
		try:
			fper, tper, thresholds = roc_curve(y_test, probs[i]) 
			plt.plot(fper, tper, color=next(cycol), label=clf_names[i]+' = %s'%(str(round(metrics.auc(fper, tper), 2))))
			plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
		except:
			print('passing %s'%(clf_names[i]))

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend()
	plt.savefig('roc_curve.png')
	plt.close()

def get_indices(selectedlabels, labels):
	'''
	takes in an list of labels and gets back indices for these labels; useful to restructure
	arrays around the right features for heatmaps.

	selectedfeatures = string of selected features from univariate feature selection
	labels = string of all potential labels in right order
	'''
	indices=list()
	for i in range(len(selectedlabels)):
		indices.append(labels.index(selectedlabels[i]))
	return indices 

def restructure_features(selectedlabels, features, labels):
	# get top 20 features instead of all features for the heatmaps
	# get top 20 feature indices
	indices=get_indices(selectedlabels, labels)

	# now that we have the indices, only select these indices in all numpy array features
	newfeatures=list()
	for i in range(len(features)):
		feature=dict()
		for j in range(len(indices)):
			feature[selectedlabels[j]]=features[i][indices[j]]
		newfeatures.append(feature)
	newfeatures=pd.DataFrame(newfeatures)
	newfeatures.to_csv('data.csv',index=False)
	return newfeatures, selectedlabels

def visualize_features(classes, problem_type, curdir, default_features, balance_data, test_size):

	# make features into label encoder here
	features, feature_labels, class_labels = get_features(classes, problem_type, default_features, balance_data)

	# now preprocess features for all the other plots
	os.chdir(curdir)
	le = preprocessing.LabelEncoder()
	le.fit(class_labels)
	tclass_labels = le.transform(class_labels)

	# process features to help with clustering
	se=preprocessing.StandardScaler()
	t_features=se.fit_transform(features)

	X_train, X_test, y_train, y_test = train_test_split(features, tclass_labels, test_size=test_size, random_state=42)

	# print(len(features))
	# print(len(feature_labels))
	# print(len(class_labels))
	# print(class_labels)
	
	# GET TRAINING DATA DURING MODELING PROCESS
	##################################
	# get filename
	# csvfile=''
	# print(classes)
	# for i in range(len(classes)):
	# 	csvfile=csvfile+classes[i]+'_'

	# get training and testing data for later
	# try:
		# print('loading training files...')
		# X_train=pd.read_csv(prev_dir(curdir)+'/models/'+csvfile+'train.csv')
		# y_train=X_train['class_']
		# X_train.drop(['class_'], axis=1)
		# X_test=pd.read_csv(prev_dir(curdir)+'/models/'+csvfile+'test.csv')
		# y_test=X_test['class_']
		# X_test.drop(['class_'], axis=1)
		# y_train=le.inverse_transform(y_train)
		# y_test=le.inverse_transform(y_test)
	# except:
	# print('error loading in training files, making new test data')

	
	# Visualize each class (quick plot)
	##################################
	visualization_dir='visualization_session'
	try:
		os.mkdir(visualization_dir)
		os.chdir(visualization_dir)
	except:
		shutil.rmtree(visualization_dir)
		os.mkdir(visualization_dir)
		os.chdir(visualization_dir)

	objects = tuple(set(class_labels))
	y_pos = np.arange(len(objects))
	performance=list()
	for i in range(len(objects)):
		performance.append(class_labels.count(objects[i]))

	plt.bar(y_pos, performance, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.xticks(rotation=90)
	plt.title('Counts per class')
	plt.ylabel('Count')
	plt.xlabel('Class')
	plt.tight_layout()
	plt.savefig('classes.png')
	plt.close()

	# set current directory
	curdir=os.getcwd()

	# ##################################
	# # CLUSTERING!!!
	# ##################################

	##################################
	# Manifold type options 
	##################################
	'''
		"lle"
		Locally Linear Embedding (LLE) uses many local linear decompositions to preserve globally non-linear structures.
		"ltsa"
		LTSA LLE: local tangent space alignment is similar to LLE in that it uses locality to preserve neighborhood distances.
		"hessian"
		Hessian LLE an LLE regularization method that applies a hessian-based quadratic form at each neighborhood
		"modified"
		Modified LLE applies a regularization parameter to LLE.
		"isomap"
		Isomap seeks a lower dimensional embedding that maintains geometric distances between each instance.
		"mds"
		MDS: multi-dimensional scaling uses similarity to plot points that are near to each other close in the embedding.
		"spectral"
		Spectral Embedding a discrete approximation of the low dimensional manifold using a graph representation.
		"tsne" (default)
		t-SNE: converts the similarity of points into probabilities then uses those probabilities to create an embedding.
	'''
	os.mkdir('clustering')
	os.chdir('clustering')

	# tSNE 
	plt.figure()
	viz = Manifold(manifold="tsne", classes=set(classes))
	viz.fit_transform(np.array(features), tclass_labels)
	viz.poof(outpath="tsne.png")   
	plt.close()
	# os.system('open tsne.png')
	# viz.show()

	# PCA
	plt.figure()
	visualizer = PCADecomposition(scale=True, classes=set(classes))
	visualizer.fit_transform(np.array(features), tclass_labels)
	visualizer.poof(outpath="pca.png") 
	plt.close()
	# os.system('open pca.png')

	# spectral embedding
	plt.figure()
	viz = Manifold(manifold="spectral", classes=set(classes))
	viz.fit_transform(np.array(features), tclass_labels)
	viz.poof(outpath="spectral.png")   
	plt.close()

	# lle embedding
	plt.figure()
	viz = Manifold(manifold="lle", classes=set(classes))
	viz.fit_transform(np.array(features), tclass_labels)
	viz.poof(outpath="lle.png")   
	plt.close()

	# ltsa
	# plt.figure()
	# viz = Manifold(manifold="ltsa", classes=set(classes))
	# viz.fit_transform(np.array(features), tclass_labels)
	# viz.poof(outpath="ltsa.png")   
	# plt.close()

	# hessian
	# plt.figure()
	# viz = Manifold(manifold="hessian", method='dense', classes=set(classes))
	# viz.fit_transform(np.array(features), tclass_labels)
	# viz.poof(outpath="hessian.png")   
	# plt.close()

	# modified
	plt.figure()
	viz = Manifold(manifold="modified", classes=set(classes))
	viz.fit_transform(np.array(features), tclass_labels)
	viz.poof(outpath="modified.png")   
	plt.close()

	# isomap
	plt.figure()
	viz = Manifold(manifold="isomap", classes=set(classes))
	viz.fit_transform(np.array(features), tclass_labels)
	viz.poof(outpath="isomap.png")   
	plt.close()

	# mds
	plt.figure()
	viz = Manifold(manifold="mds", classes=set(classes))
	viz.fit_transform(np.array(features), tclass_labels)
	viz.poof(outpath="mds.png")   
	plt.close()

	# spectral
	plt.figure()
	viz = Manifold(manifold="spectral", classes=set(classes))
	viz.fit_transform(np.array(features), tclass_labels)
	viz.poof(outpath="spectral.png")   
	plt.close()

	# UMAP embedding
	plt.figure()
	umap = UMAPVisualizer(metric='cosine', classes=set(classes), title="UMAP embedding")
	umap.fit_transform(np.array(features), class_labels)
	umap.poof(outpath="umap.png") 
	plt.close()

	# alternative UMAP
	# import umap.plot
	# plt.figure()
	# mapper = umap.UMAP().fit(np.array(features))
	# fig=umap.plot.points(mapper, labels=np.array(tclass_labels))
	# fig = fig.get_figure()
	# fig.tight_layout()
	# fig.savefig('umap2.png')
	# plt.close(fig)

	#################################
	# 	  FEATURE RANKING!!
	#################################
	os.chdir(curdir)
	os.mkdir('feature_ranking')
	os.chdir('feature_ranking')

	# You can get the feature importance of each feature of your dataset 
	# by using the feature importance property of the model.
	plt.figure(figsize=(12,12))
	model = ExtraTreesClassifier()
	model.fit(np.array(features),tclass_labels)
	# print(model.feature_importances_)
	feat_importances = pd.Series(model.feature_importances_, index=feature_labels[0])
	feat_importances.nlargest(20).plot(kind='barh')
	plt.title('Feature importances (ExtraTrees)', size=16)
	plt.title('Feature importances with %s features'%(str(len(features[0]))))
	plt.tight_layout()
	plt.savefig('feature_importance.png')
	plt.close()
	# os.system('open feature_importance.png')

	# get selected labels for top 20 features
	selectedlabels=list(dict(feat_importances.nlargest(20)))
	new_features, new_labels = restructure_features(selectedlabels, t_features, feature_labels[0])
	new_features_, new_labels_ = restructure_features(selectedlabels, features, feature_labels[0])

	# Shapiro rank algorithm (1D)
	plt.figure(figsize=(28,12))
	visualizer = Rank1D(algorithm='shapiro', classes=set(classes), features=new_labels)
	visualizer.fit(np.array(new_features), tclass_labels)
	visualizer.transform(np.array(new_features))
	# plt.tight_layout()
	visualizer.poof(outpath="shapiro.png")
	plt.title('Shapiro plot (top 20 features)', size=16)
	plt.close()
	# os.system('open shapiro.png')
	# visualizer.show()   

	# pearson ranking algorithm (2D)
	plt.figure(figsize=(12,12))
	visualizer = Rank2D(algorithm='pearson', classes=set(classes), features=new_labels)
	visualizer.fit(np.array(new_features), tclass_labels)
	visualizer.transform(np.array(new_features))
	plt.tight_layout()
	visualizer.poof(outpath="pearson.png")
	plt.title('Pearson ranking plot (top 20 features)', size=16)
	plt.close()
	# os.system('open pearson.png')
	# visualizer.show()   

	# feature importances with top 20 features for Lasso
	plt.figure(figsize=(12,12))
	viz = FeatureImportances(Lasso(), labels=new_labels_)
	viz.fit(np.array(new_features_), tclass_labels)
	plt.tight_layout()
	viz.poof(outpath="lasso.png")
	plt.close()

	# correlation plots with feature removal if corr > 0.90
	# https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf

	# now remove correlated features
	# --> p values
	# --> https://towardsdatascience.com/the-next-level-of-data-visualization-in-python-dd6e99039d5e / https://github.com/WillKoehrsen/Data-Analysis/blob/master/plotly/Plotly%20Whirlwind%20Introduction.ipynb- plotly for correlation heatmap and scatterplot matrix
	# --> https://seaborn.pydata.org/tutorial/distributions.html
	data=new_features
	corr = data.corr()

	plt.figure(figsize=(12,12))
	fig=sns.heatmap(corr)
	fig = fig.get_figure()
	plt.title('Heatmap with correlated features (top 20 features)', size=16)
	fig.tight_layout()
	fig.savefig('heatmap.png')
	plt.close(fig)

	columns = np.full((corr.shape[0],), True, dtype=bool)
	for i in range(corr.shape[0]):
	    for j in range(i+1, corr.shape[0]):
	        if corr.iloc[i,j] >= 0.9:
	            if columns[j]:
	                columns[j] = False
	selected_columns = data.columns[columns]
	data = data[selected_columns]
	corr=data.corr()

	plt.figure(figsize=(12,12))
	fig=sns.heatmap(corr)
	fig = fig.get_figure()
	plt.title('Heatmap without correlated features (top 20 features)', size=16)
	fig.tight_layout()
	fig.savefig('heatmap_clean.png')
	plt.close(fig)

	# radviz
	# Instantiate the visualizer
	plt.figure(figsize=(12,12))
	visualizer = RadViz(classes=classes, features=new_labels)
	visualizer.fit(np.array(new_features), tclass_labels)           
	visualizer.transform(np.array(new_features))  
	visualizer.poof(outpath="radviz.png")     
	visualizer.show()         
	plt.close()     

	# feature correlation plot
	plt.figure(figsize=(28,12))
	visualizer = feature_correlation(np.array(new_features), tclass_labels, labels=new_labels)
	visualizer.poof(outpath="correlation.png") 
	visualizer.show()
	plt.tight_layout()
	plt.close()

	os.mkdir('feature_plots')
	os.chdir('feature_plots')

	newdata=new_features_
	newdata['classes']=class_labels

	for j in range(len(new_labels_)):
		fig=sns.violinplot(x=newdata['classes'], y=newdata[new_labels_[j]])
		fig = fig.get_figure()
		fig.tight_layout()
		fig.savefig('%s_%s.png'%(str(j), new_labels_[j]))
		plt.close(fig)

	os.mkdir('feature_plots_transformed')
	os.chdir('feature_plots_transformed')

	newdata=new_features
	newdata['classes']=class_labels

	for j in range(len(new_labels)):
		fig=sns.violinplot(x=newdata['classes'], y=newdata[new_labels[j]])
		fig = fig.get_figure()
		fig.tight_layout()
		fig.savefig('%s_%s.png'%(str(j), new_labels[j]))
		plt.close(fig)
		
	##################################################
	# PRECISION-RECALL CURVES
	##################################################

	os.chdir(curdir)
	os.mkdir('model_selection')
	os.chdir('model_selection')

	plt.figure()
	visualizer = precision_recall_curve(GaussianNB(), np.array(features), tclass_labels)
	visualizer.poof(outpath="precision-recall.png")
	plt.close()

	plt.figure()
	visualizer = roc_auc(LogisticRegression(), np.array(features), tclass_labels)
	visualizer.poof(outpath="roc_curve_train.png")
	plt.close()

	plt.figure()
	visualizer = discrimination_threshold(
	    LogisticRegression(multi_class="auto", solver="liblinear"), np.array(features), tclass_labels)
	visualizer.poof(outpath="thresholds.png")
	plt.close()

	plt.figure()
	visualizer = residuals_plot(
	    Ridge(), np.array(features), tclass_labels, train_color="maroon", test_color="gold"
	)
	visualizer.poof(outpath="residuals.png")
	plt.close()

	plt.figure()
	visualizer = prediction_error(Lasso(), np.array(features), tclass_labels)
	visualizer.poof(outpath='prediction_error.png')
	plt.close()

	# outlier detection
	plt.figure()
	visualizer = cooks_distance(np.array(features), tclass_labels, draw_threshold=True, linefmt="C0-", markerfmt=",")
	visualizer.poof(outpath='outliers.png')
	plt.close()

	# cluster numbers
	plt.figure()
	visualizer = silhouette_visualizer(KMeans(len(set(tclass_labels)), random_state=42), np.array(features))
	visualizer.poof(outpath='siloutte.png')
	plt.close()

	# cluster distance
	plt.figure()
	visualizer = intercluster_distance(KMeans(len(set(tclass_labels)), random_state=777), np.array(features))
	visualizer.poof(outpath='cluster_distance.png')
	plt.close()

	# plot percentile of features plot with SVM to see which percentile for features is optimal
	features=preprocessing.MinMaxScaler().fit_transform(features)
	clf = Pipeline([('anova', SelectPercentile(chi2)),
	                ('scaler', StandardScaler()),
	                ('logr', LogisticRegression())])
	score_means = list()
	score_stds = list()
	percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100)

	for percentile in percentiles:
	    clf.set_params(anova__percentile=percentile)
	    this_scores = cross_val_score(clf, np.array(features), class_labels)
	    score_means.append(this_scores.mean())
	    score_stds.append(this_scores.std())

	plt.errorbar(percentiles, score_means, np.array(score_stds))
	plt.title('Performance of the LogisticRegression-Anova varying the percent features selected')
	plt.xticks(np.linspace(0, 100, 11, endpoint=True))
	plt.xlabel('Percentile')
	plt.ylabel('Accuracy Score')
	plt.axis('tight')
	plt.savefig('logr_percentile_plot.png')
	plt.close()

	# get PCA
	pca = PCA(random_state=1)
	pca.fit(X_train)
	skplt.decomposition.plot_pca_component_variance(pca)
	plt.savefig('pca_explained_variance.png')
	plt.close()

	# estimators
	rf = RandomForestClassifier()
	skplt.estimators.plot_learning_curve(rf, X_train, y_train)
	plt.title('Learning Curve (Random Forest)')
	plt.savefig('learning_curve.png')
	plt.close()

	# elbow plot
	kmeans = KMeans(random_state=1)
	skplt.cluster.plot_elbow_curve(kmeans, X_train, cluster_ranges=range(1, 30), title='Elbow plot (KMeans clustering)')
	plt.savefig('elbow.png')
	plt.close()

	# KS statistic (only if 2 classes)
	lr = LogisticRegression()
	lr = lr.fit(X_train, y_train)
	y_probas = lr.predict_proba(X_test)
	skplt.metrics.plot_ks_statistic(y_test, y_probas)
	plt.savefig('ks.png')
	plt.close()

	# precision-recall
	nb = GaussianNB()
	nb.fit(X_train, y_train)
	y_probas = nb.predict_proba(X_test)
	skplt.metrics.plot_precision_recall(y_test, y_probas)
	plt.tight_layout()
	plt.savefig('precision-recall.png')
	plt.close()

	## plot calibration curve 
	rf = RandomForestClassifier()
	lr = LogisticRegression()
	nb = GaussianNB()
	svm = LinearSVC()
	dt = DecisionTreeClassifier(random_state=0)
	ab = AdaBoostClassifier(n_estimators=100)
	gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
	knn = KNeighborsClassifier(n_neighbors=7)

	rf_probas = rf.fit(X_train, y_train).predict_proba(X_test)
	lr_probas = lr.fit(X_train, y_train).predict_proba(X_test)
	nb_probas = nb.fit(X_train, y_train).predict_proba(X_test)
	# svm_scores = svm.fit(X_train, y_train).predict_proba(X_test)
	dt_scores= dt.fit(X_train, y_train).predict_proba(X_test)
	ab_scores= ab.fit(X_train, y_train).predict_proba(X_test)
	gb_scores= gb.fit(X_train, y_train).predict_proba(X_test)
	knn_scores= knn.fit(X_train, y_train).predict_proba(X_test)

	probas_list = [rf_probas, lr_probas, nb_probas, # svm_scores,
				   dt_scores, ab_scores, gb_scores, knn_scores]

	clf_names = ['Random Forest', 'Logistic Regression', 'Gaussian NB', # 'SVM',
				 'Decision Tree', 'Adaboost', 'Gradient Boost', 'KNN']

	skplt.metrics.plot_calibration_curve(y_test,probas_list, clf_names)
	plt.savefig('calibration.png')
	plt.tight_layout()
	plt.close()

	# pick classifier type by ROC (without optimization)
	probs = [rf_probas[:, 1], lr_probas[:, 1], nb_probas[:, 1], # svm_scores[:, 1],
			 dt_scores[:, 1], ab_scores[:, 1], gb_scores[:, 1], knn_scores[:, 1]]

	plot_roc_curve(y_test, probs, clf_names)
	# more elaborate ROC example with CV = 5 fold 
	# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py

	os.chdir(curdir)

	return ''

# get current directory 
curdir=os.getcwd()
basedir=prev_dir(curdir)
os.chdir(basedir+'/train_dir')
problem_type=''
try:
	problem_type=sys.argv[1]
except:
	# added this in for the CLI
	while problem_type not in ['audio', 'text', 'image', 'video', 'csv']:
		problem_type = input('what is the problem that you are going after (e.g. "audio", "text", "image","video","csv")\n')
	print(problem_type)
classes=get_classes()

# get default features to use in visualization
settings=json.load(open(basedir+'/settings.json'))
if problem_type=='audio':
	default_features=settings['default_audio_features']
elif problem_type=='text':
	default_features=settings['default_text_features']
elif problem_type=='image':
	default_features=settings['default_image_features']
elif problem_type=='video':
	default_features=settings['default_video_features']
elif problem_type=='csv':
	default_features=settings['default_csv_features']

balance_data=settings['balance_data']
test_size=settings['test_size']

visualize_features(classes, problem_type, curdir, default_features, balance_data, test_size)
