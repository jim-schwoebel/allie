'''
visualize.py
Takes in a folder or set of folders of featurized files and outputs
visualizations to look deeper at the data.
This is often useful as a precursor before building machine learning 
models to uncover relationships in the data.
Note that this automatically happens as part of the modeling process
if visualize==True in settings.
ML models: https://medium.com/analytics-vidhya/how-to-visualize-anything-in-machine-learning-using-yellowbrick-and-mlxtend-39c45e1e9e9f
'''
import os, sys, json, time
from tqdm import tqdm
from yellowbrick.features import Rank1D, Rank2D, Manifold
from yellowbrick.features.pca import PCADecomposition
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import pandas as pd

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

	return classes

def get_features(classes, problem_type, default_features):

	features=list()
	feature_labels=list()
	class_labels=list()
	curdir=os.getcwd()

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
			g=json.load(open(jsonfiles[j]))
			feature_=list()
			label_=list()
			try:
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

def visualize_features(classes, problem_type, curdir, default_features):

	features, feature_labels, class_labels = get_features(classes, problem_type, default_features)
	print(features)
	os.chdir(curdir)
	le = preprocessing.LabelEncoder()
	le.fit(class_labels)
	tclass_labels = le.transform(class_labels)

	print(len(features))
	print(len(feature_labels))
	print(len(class_labels))
	print(class_labels)

	# Visualize each class (quick plot)
	##################################
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

	# Shapiro rank algorithm (1D)
	plt.figure()
	visualizer = Rank1D(algorithm='shapiro', classes=set(classes))
	visualizer.fit(np.array(features), tclass_labels)
	visualizer.transform(np.array(features))
	visualizer.poof(outpath="shapiro.png")
	plt.close()
	# os.system('open shapiro.png')
	# visualizer.show()   

	# pearson ranking algorithm (2D)
	plt.figure()
	visualizer = Rank2D(algorithm='pearson', classes=set(classes))
	visualizer.fit(np.array(features), tclass_labels)
	visualizer.transform(np.array(features))
	visualizer.poof(outpath="pearson.png")
	plt.close()
	# os.system('open pearson.png')
	# visualizer.show()   

	# You can get the feature importance of each feature of your dataset 
	# by using the feature importance property of the model.
	plt.figure()
	model = ExtraTreesClassifier()
	model.fit(np.array(features),tclass_labels)
	print(model.feature_importances_)
	feat_importances = pd.Series(model.feature_importances_, index=feature_labels[0])
	feat_importances.nlargest(10).plot(kind='barh')
	plt.tight_layout()
	plt.savefig('feature_importance.png')
	os.system('open feature_importance.png')

	return ''

# get current directory 
curdir=os.getcwd()
basedir=prev_dir(curdir)
os.chdir(basedir+'/train_dir')
problem_type=sys.argv[1]
# 	plt.ylabel('Usage')
# 	plt.title('Programming language usage')
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

visualize_features(classes, problem_type, curdir, default_features)
