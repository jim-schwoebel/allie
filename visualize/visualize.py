'''
visualize.py

Takes in a folder or set of folders of featurized files and outputs
visualizations to look deeper at the data.

This is often useful as a precursor before building machine learning 
models to uncover relationships in the data.

Note that this automatically happens as part of the modeling process
if visualize==True in settings.
'''
import os, sys, json
from tqdm import tqdm

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

def get_features(classes, problem_type):

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
			for k in range(len(feature_list)):
				feature_=feature_+g['features'][problem_type][feature_list[k]]['features']
				label_=g['features'][problem_type][feature_list[k]]['labels']

			# quick quality check to only add to list if the feature_labels match in length the features_
			if len(feature_) == len(label_):
				features.append(feature_)
				feature_labels.append(label_)
				class_labels.append(classes[i])


	return features, feature_labels, class_labels 

def visualize_features(classes, problem_type):

	features, feature_labels, class_labels = get_features(classes, problem_type)
	print(len(features))
	print(len(feature_labels))
	print(len(class_labels))
	print(class_labels)

	if problem_type=='audio':
		# yellowbrick
		pass

	elif problem_type=='text':
		# tSNE
		pass

	elif problem_type=='image':
		# other plots
		pass 

	elif problem_type=='video':
		# video plots
		pass

	elif problem_type=='csv':
		# class labels
		pass

	return ''

# get current directory 
curdir=os.getcwd()
basedir=prev_dir(curdir)
os.chdir(basedir+'/train_dir')
problem_type=sys.argv[1]
print(problem_type)
classes=get_classes()

visualize_features(classes, problem_type)
