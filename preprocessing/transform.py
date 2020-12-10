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

______                                           _             
| ___ \                                         (_)            
| |_/ / __ ___ _ __  _ __ ___   ___ ___  ___ ___ _ _ __   __ _ 
|  __/ '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
| |  | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
\_|  |_|  \___| .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, |
              | |                                         __/ |
              |_|                                        |___/ 
  ___  ______ _____ 
 / _ \ | ___ \_   _|
/ /_\ \| |_/ / | |  
|  _  ||  __/  | |  
| | | || |    _| |_ 
\_| |_/\_|    \___/ 

Make feature transformations as a part of Allie's preprocessing API. 

Transformation are done in this order:

Feature scalers --> reduce dimensions --> select features.

A --> A`--> A`` --> A```

------
Usage: python3 transform.py audio c gender males females
------

sys.argv[1] - 'audio' - the sample type of file preprocessed by the transformer	
['audio', 'text', 'image', 'video', 'csv']

sys.argv[2] - 'c' - classification or regression problems 
['c', 'r']

sys.argv[3] - 'gender' - the common name for the transformer 
can be any string (e.g. 'gender')

sys.argv[4], sys.argv[5], sys.argv[n]	
'males'	classes that you seek to model in the train_dir folder	any string folder name

'''

import json, os, sys, time, pickle
os.system('pip3 install scikit-learn==0.22.2.post1')
import numpy as np 
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
            
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

def get_class_dir():
	count=4
	classes=list()
	while True:
		try:
			class_=sys.argv[count]
			classes.append(class_)
			print(classes)
			count=count+1
		except:
			break

	classdirs=list()
	actualclasses=list()
	for i in range(len(classes)):
		if classes[i].count('/') > 0:
			# a directory 
			classdirs.append(classes[i])
			actualclasses.append(classes[i].split('/')[-1])
		else:
			classdirs.append(os.getcwd()+'/'+classes[i])
			actualclasses.append(classes[i])

	return classdirs, actualclasses

def get_features(classdirs, classes, problem_type, settings):

	# initialize lists
	features=list()
	feature_labels=list()
	class_labels=list()
	curdir=os.getcwd()

	# get defaults (if multiple feature arrays)
	default_audio_features=settings["default_audio_features"]
	default_text_features=settings["default_text_features"]
	default_image_features=settings["default_image_features"]
	default_video_features=settings["default_video_features"]
	default_csv_features=settings["default_csv_features"]

	defaults=default_audio_features+default_text_features+default_image_features+default_video_features+default_csv_features

	for i in range(len(classes)):
		classname=class_labels
		print('----------LOADING %s----------'%(classes[i].upper()))
		os.chdir(classdirs[i])
		listdir=os.listdir()
		jsonfiles=list()

		for j in range(len(listdir)):
			if listdir[j].endswith('.json'):
				jsonfiles.append(listdir[j])

		g=json.load(open(jsonfiles[0]))
		feature_list=list(g['features'][problem_type])

		for j in tqdm(range(len(jsonfiles))):
			try:
				g=json.load(open(jsonfiles[j]))
				feature_=list()
				label_=list()
				try:
					for k in range(len(feature_list)):
						if feature_list[k] in defaults:
							feature_=feature_+g['features'][problem_type][feature_list[k]]['features']
							label_=label_+g['features'][problem_type][feature_list[k]]['labels']

					# quick quality check to only add to list if the feature_labels match in length the features_
					if len(feature_) == len(label_):
						features.append(feature_)
						feature_labels.append(label_)
						class_labels.append(classes[i])
				except:
					print('error loading feature embedding: %s'%(feature_list[k].upper()))
			except:
				print('error loading %s -> %s'%(classes[i].upper(), jsonfiles[j]))


	return features, feature_labels, class_labels 

################################################
##	    		Load main settings    		  ##
################################################

# directory=sys.argv[1]
basedir=os.getcwd()
settingsdir=prev_dir(basedir)
print(settingsdir)
settings=json.load(open(settingsdir+'/settings.json'))

# get all the important settings for the transformations 
scale_features=settings['scale_features']
reduce_dimensions=settings['reduce_dimensions']
select_features=settings['select_features']
default_scalers=settings['default_scaler']
default_reducers=settings['default_dimensionality_reducer']
default_selectors=settings['default_feature_selector']

print(scale_features)
print(reduce_dimensions)
print(select_features)
print(default_scalers)
print(default_reducers)
print(default_selectors)

os.chdir(basedir)

################################################
##	    	Now go featurize!                 ##
################################################

# get current directory 
curdir=os.getcwd()
basedir=prev_dir(curdir)
os.chdir(basedir+'/train_dir')
problem_type=sys.argv[1] #audio, text, image, video, csv
train_type=sys.argv[2] #c = classification, r=regression

if train_type == 'c':
	common_name=sys.argv[3] #common_name = 'gender'
	if problem_type != 'csv':
		classdirs, classes=get_class_dir()
		features, feature_labels, class_labels = get_features(classdirs, classes, problem_type, settings)
	elif problem_type == 'csv':
		class_='class_'
		classes=['class_']
		os.chdir(basedir+'/models/')
		g=pd.read_csv(common_name+'_all.csv')
		features=g.drop(['class_'], axis=1)
		feature_labels=list(features)
		features_=list()
		for i in range(len(features)):
			features_.append(list(features.iloc[i,:]))
		features=features_
		class_labels=list()
		for i in range(len(features)):
			class_labels.append(class_)
		os.chdir(curdir)
	X_train, X_test, y_train, y_test = train_test_split(features, class_labels, train_size=0.90, test_size=0.10)
	print(features[0])
	print(feature_labels[0])

elif train_type == 'r':
	# only 1 class here 
	target=[sys.argv[3]]
	spreadsheet=sys.argv[4]
	spreadsheet_dir=sys.argv[5]
	print(target)
	print(spreadsheet)
	print(spreadsheet_dir)
	common_name=sys.argv[6] #common_name = 'gender'
	os.chdir(spreadsheet_dir)
	data=pd.read_csv(spreadsheet)
	features=np.array(data.drop(columns=target, axis=1))
	feature_labels=list(features)
	class_labels=np.array(data.pop(target[0]))

	print(features)
	print(feature_labels)
	print(class_labels)

	X_train, X_test, y_train, y_test = train_test_split(features, class_labels, train_size=0.90, test_size=0.10)

# create a scikit-learn pipeline
feature_num=settings['feature_number']
component_num=settings['dimension_number']

estimators = []
os.chdir(basedir+'/train_dir')

# for i in range(len(X_train)):
	# if len(X_train[i]) != len(X_train[0]):
		# print(X_train[i])

################################################
##	    	    Scale features               ##
################################################
if scale_features == True:
	import feature_scale as fsc_
	for i in range(len(default_scalers)):
		feature_scaler=default_scalers[i]
		print(feature_scaler.upper())
		scaler_model=fsc_.feature_scale(feature_scaler, X_train, y_train)
		# print(len(scaler_model))
		estimators.append((feature_scaler, scaler_model))

################################################
##	    	   Reduce dimensions              ##
################################################
if reduce_dimensions == True:
	import feature_reduce as fre_
	for i in range(len(default_reducers)):
		feature_reducer=default_reducers[i]
		print(feature_reducer.upper()+' - %s features'%(str(component_num)))
		dimension_model=fre_.feature_reduce(feature_reducer, X_train, y_train, component_num)
		# print(len(dimension_model))
		estimators.append((feature_reducer, dimension_model))

################################################
##	    	   Feature selection              ##
################################################
if select_features == True:
	import feature_select as fse_
	for i in range(len(default_selectors)):
		feature_selector=default_selectors[i]
		print(feature_selector.upper()+' - %s features'%(str(feature_num)))
		selection_model=fse_.feature_select(feature_selector, X_train, y_train, feature_num)
		estimators.append((feature_selector, selection_model))

print(estimators)
model=Pipeline(estimators)

# make all train and test data into binary labels
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
if train_type == 'c':
	le=preprocessing.LabelEncoder()
	le.fit(y_train)
	y_train=le.transform(y_train)
	y_test=le.transform(y_test)

'''
>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"])
array([2, 2, 1]...)
>>> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']
'''

model=model.fit(X_train, y_train)
print(len(X_test))
X_test=model.transform(X_test)
print(len(X_test))

print('transformed training size')
print(X_test[0])

# pickle me timbers
os.chdir(curdir)
print(os.getcwd())

try:
	os.chdir('%s_transformer'%(problem_type))
except:
	os.mkdir('%s_transformer'%(problem_type))
	os.chdir('%s_transformer'%(problem_type))

# get filename / create a unique file name
filename=train_type+'_'+common_name

# only add names in if True 
if scale_features == True:
	for i in range(len(default_scalers)):
		filename=filename+'_'+default_scalers[i]
if reduce_dimensions == True:
	for i in range(len(default_reducers)):
		filename=filename+'_'+default_reducers[i]
if select_features == True:
	for i in range(len(default_selectors)):
		filename=filename+'_'+default_selectors[i]

model_file=filename+'.pickle'
le_file=filename+'_encoder.pickle'
json_file=filename+'.json'

# create model
modelfile=open(model_file,'wb')
pickle.dump(model, modelfile)
modelfile.close()

# save encoder (to decode classes into the future)
if train_type == 'c':
	modelfile=open(le_file,'wb')
	pickle.dump(le, modelfile)
	modelfile.close()


# write json file 
if train_type=='c':
	data={'estimators': str(estimators),
		  'settings': settings,
		  'classes': np.array(list(set(y_test))).tolist(),
		  'sample input X': X_train[0],
		  'sample input Y': int(y_train[0]),
		  'sample transformed X': X_test[0].tolist(),
		  'sample transformed y': int(y_train[0]),
		 }
else:
	data={'estimators': str(estimators),
		  'settings': settings,
		  'classes': np.array(list(set(y_test))).tolist(),
		  'sample input X': X_train[0].tolist(),
		  'sample input Y': float(y_train[0]),
		  'sample transformed X': X_test[0].tolist(),
		  'sample transformed y': float(y_train[0]),
		 }

# for testing purposes
# data_list=list(data)
# for i in range(len(data_list)):
# 	print(data_list[i])
# 	print(type(data[data_list[i]]))
# 	if str(type(data[data_list[i]])) == "<class 'list'>":
# 		for j in range(len(data[data_list[i]])):
# 			print(type(data[data_list[i]][j]))

jsonfile=open(json_file,'w')
json.dump(data,jsonfile)
jsonfile.close()
