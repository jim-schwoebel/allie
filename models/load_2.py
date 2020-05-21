'''
Load all models and make model predictions intelligently based on files 
in the load_dir folder.

Note that the general flow here is 

1. load settings.json
2. featurize all the files [audio, text, image, video files]
3. apply all machine learning models available
'''

import os, json

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

def featurize(features_dir, load_dir, model_dir):
	# featurize by directories
	if model_dir=='audio_models':
		os.chdir(features_dir+'/audio_features')
	elif model_dir=='text_features':	
		os.chdir(features_dir+'/text_features')
	elif model_dir=='image_features':
		os.chdir(features_dir+'/image_features')
	elif model_dir=='video_features':	
		os.chdir(features_dir+'/video_features')
	elif model_dir=='csv_features':
		os.chdir(features_dir+'/csv_features')
	
	os.system('python3 featurize.py %s'%(load_dir))

def find_files(model_dir):

	if model_dir == 'audio_models':
		listdir=os.listdir()
		for i in range(len(listdir)):
			jsonfile=listdir[i][0:-4]+'.json'
			if listdir[i].endswith('.wav') and jsonfile in listdir:
				jsonfiles.append(jsonfile)

	elif model_dir == 'text_models':
		listdir=os.listdir()
		for i in range(len(listdir)):
			jsonfile=listdir[i][0:-4]+'.json'
			if listdir[i].endswith('.txt') and jsonfile in listdir:
				jsonfiles.append(jsonfile)

	elif model_dir == 'image_models':
		listdir=os.listdir()
		for i in range(len(listdir)):
			jsonfile=listdir[i][0:-4]+'.json'
			if listdir[i].endswith('.png') and jsonfile in listdir:
				jsonfiles.append(jsonfile)

	elif model_dir == 'video_models':
		listdir=os.listdir()
		for i in range(len(listdir)):
			jsonfile=listdir[i][0:-4]+'.json'
			if listdir[i].endswith('.mp4') and jsonfile in listdir:
				jsonfiles.append(jsonfile)

	elif model_dir =='csv_models':

		listdir=os.listdir()
		for i in range(len(listdir)):
			jsonfile=listdir[i][0:-4]+'.json'
			if listdir[i].endswith('.csv') and jsonfile in listdir:
				jsonfiles.append(jsonfile)

	else:
		jsonfiles=[]

	return jsonfiles

def make_prediction(transformer, clf, modeltype, jsonfiles):
	'''
	get the metrics associated iwth a classification and regression problem
	and output a .JSON file with the training session.
	'''

	for k in range(len(jsonfiles)):

		g=json.load(open(jsonfiles[k]))

		metrics_=dict()
		y_true=y_test

		if modeltype not in ['autogluon', 'autokeras', 'autopytorch', 'alphapy', 'atm', 'keras', 'devol', 'ludwig', 'safe', 'neuraxle']:
			y_pred=clf.predict(features)

		elif modeltype=='alphapy':
			# go to the right folder 
			curdir=os.getcwd()
			print(os.listdir())
			os.chdir(common_name+'_alphapy_session')
			alphapy_dir=os.getcwd()
			os.chdir('input')
			os.rename('test.csv', 'predict.csv')
			os.chdir(alphapy_dir)
			os.system('alphapy --predict')
			os.chdir('output')
			listdir=os.listdir()
			for k in range(len(listdir)):
				if listdir[k].startswith('predictions'):
					csvfile=listdir[k]
			y_pred=pd.read_csv(csvfile)['prediction']
			os.chdir(curdir)

		elif modeltpe == 'autogluon':
			from autogluon import TabularPrediction as task
			test_data=test_data.drop(labels=['class'],axis=1)
			y_pred=clf.predict(test_data)

		elif modeltype == 'autokeras':
			y_pred=clf.predict(features).flatten()

		elif modeltype == 'autopytorch':
			y_pred=clf.predict(features).flatten()

		elif modeltype == 'atm':
			curdir=os.getcwd()
			os.chdir('atm_temp')
			data = pd.read_csv('test.csv').drop(labels=['class_'], axis=1)
			y_pred = clf.predict(data)
			os.chdir(curdir)

		elif modeltype == 'ludwig':
			data=pd.read_csv('test.csv').drop(labels=['class_'], axis=1)
			pred=clf.predict(data)['class__predictions']
			y_pred=np.array(list(pred), dtype=np.int64)

		elif modeltype== 'devol':
			features=features.reshape(features.shape+ (1,)+ (1,))
			y_pred=clf.predict_classes(features).flatten()

		elif modeltype ='keras':
			if mtype == 'c':
			    y_pred=clf.predict_classes(features).flatten()
			elif mtype == 'r':
				y_pred=clf.predict(feaures).flatten()

		elif modeltype =='neuraxle':
			y_pred=clf.transform(features)

		elif modeltype=='safe':
			# have to make into a pandas dataframe
			test_data=pd.read_csv('test.csv').drop(columns=['class_'], axis=1)
			y_pred=clf.predict(test_data)

		# update model in schema

	return y_pred

def load_model(folder_name):

	listdir=os.listdir()

	# load in a transform if necessary
	if listdir[i].endswith('transform.pickle') in listdir:
		transform_=open(listdir[i],'rb')
		transform=pickle.load(transform_name)
		transform_.close()
	else:
		transform=''

	jsonfile=open(folder_name+'.json')
	g=json.load()
	jsonfile.close()

	# get model name
	model_name=g['model name']
	model_type=g['model type']

	# load model for getting metrics
	if model_type not in ['alphapy', 'atm', 'autokeras', 'autopytorch', 'ludwig', 'keras', 'devol']:
		loadmodel=open(modelname, 'rb')
		clf=pickle.load(loadmodel)
		loadmodel.close()
	elif model_type == 'atm':
		from atm import Model
		clf=Model.load(modelname)
	elif model_type == 'autokeras':
		import tensorflow as tf
		import autokeras as ak
		clf = pickle.load(open(modelname, 'rb'))
	elif model_type=='autopytorch':
		import torch
		clf=torch.load(modelname)
	elif model_type == 'ludwig':
		from ludwig.api import LudwigModel
		clf=LudwigModel.load('ludwig_files/experiment_run/model/')
	elif model_type in ['devol', 'keras']: 
		from keras.models import load_model
		clf = load_model(modelname)
	else: 
		clf=''

	return transformer, clf, modeltype

def find_models():

	curdir=os.getcwd()
	listdir=os.listdir()
	directories=['audio_models', 'text_models', 'image_models', 'video_models', 'csv_models']
	models_=dict()

	for i in range(len(directories)):
		model_names=list()
		try:
			os.chdir(curdir)
			os.chdir(directories[i])
			listdir=os.listdir()

			folders=list()
			for j in range(len(listdir)):
				if listdir[j].find('.') < 0:
					folders.append(listdir[j])

			curdir2=os.getcwd()

			for j in range(len(folders)):
				os.chdir(curdir2)
				os.chdir(folders[j])
				os.chdir('model')
				listdir2=os.listdir()
				jsonfile=folders[j]+'.json'
				for k in range(len(listdir2)):
					if listdir2[k] == jsonfile:
						g=json.load(open(jsonfile))
						model_names.append(jsonfile[0:-5])
		except:
			pass

		models_[directories[i]]=model_names

	print(models_)
	
	return models_

# get folders
curdir=os.getcwd()
basedir=prev_dir(curdir)
os.chdir(basedir)

# load settings 
settings=json.load(open('settings.json'))

# get the base audio, text, image, and video features from required models
default_audio_features=settings['default_audio_features']
default_text_features=settings['defualt_text_features']
default_image_features=settings['default_image_features']
default_video_features=settings['default_video_features']
default_csv_features=settings['default_csv_features']

features_dir=basedir+'/features'
model_dir=basedir+'/models'
load_dir=basedir+'/load_dir'

# find all machine learning models
os.chdir(model_dir)
models=find_models()
model_dirs=list(models)

# now that we have all the models we can begin to load all of them 
for i in range(len(model_dirs)):
	featurize(features_dir, load_dir, model_dirs[i])

# now model everything
for i in range(len(model_dirs)):
	os.chdir(model_dir)
	os.chdir(model_dirs[i])
	models_=models[model_dirs[i]]
	for j in range(len(models_)):
		os.chdir(models_[j])
		os.chdir('model')
		transformer, clf, modeltype = load_model(models_[j])
		jsonfiles=find_files(model_dirs[i])
		make_prediction(transformer, clf, modeltype, jsonfiles)
