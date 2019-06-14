'''
PLDA implementation from
https://github.com/RaviSoji/plda/blob/master/mnist_demo/mnist_demo.ipynb
'''
import os, sys, pickle, json, random, shutil
import numpy as np
import matplotlib.pyplot as plt

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

def get_folders(listdir):
	folders=list()
	for i in range(len(listdir)):
		if listdir[i].find('.') < 0:
			folders.append(listdir[i])

	return folders 

def classifyfolder(listdir):
	filetypes=list()
	for i in range(len(listdir)):
		if listdir[i].endswith(('.mp3', '.wav')):
			filetypes.append('audio')
		elif listdir[i].endswith(('.png', '.jpg')):
			filetypes.append('image')
		elif listdir[i].endswith(('.txt')):
			filetypes.append('text')
		elif listdir[i].endswith(('.mp4', '.avi')):
			filetypes.append('video')
		elif listdir[i].endswith(('.csv')):
			filetypes.append('csv')

	counts={'audio': filetypes.count('audio'),
			'image': filetypes.count('image'),
			'text': filetypes.count('text'),
			'video': filetypes.count('video'),
			'csv': filetypes.count('.csv')}

	# get back the type of folder (main file type)
	countlist=list(counts)
	countvalues=list(counts.values())
	maxvalue=max(countvalues)
	maxind=countvalues.index(maxvalue)
	return countlist[maxind]

# load the default feature set 
cur_dir = os.getcwd()
prevdir= prev_dir(cur_dir)
sys.path.append(prevdir+'/train_dir')
settings=json.load(open(prevdir+'/settings.json'))

# get all the default feature arrays 
default_audio_features=settings['default_audio_features']
default_text_features=settings['default_text_features']
default_image_features=settings['default_image_features']
default_video_features=settings['default_video_features']
default_csv_features='csv'

# prepare training and testing data (should have been already featurized) - # of classes/folders
os.chdir(prevdir+'/train_dir')

data_dir=os.getcwd()
listdir=os.listdir()
folders=get_folders(listdir)

# now assess folders by content type 
data=dict()
for i in range(len(folders)):
	os.chdir(folders[i])
	listdir=os.listdir()
	filetype=classifyfolder(listdir)
	data[folders[i]]=filetype 
	os.chdir(data_dir)

# now ask user what type of problem they are trying to solve 
problemtype=input('what problem are you solving? (1-audio, 2-text, 3-image, 4-video, 5-csv)\n')
while problemtype not in ['1','2','3','4','5']:
	print('answer not recognized...')
	problemtype=input('what problem are you solving? (1-audio, 2-text, 3-image, 4-video, 5-csv)\n')

if problemtype=='1':
	problemtype='audio'
elif problemtype=='2':
	problemtype='text'
elif problemtype=='3':
	problemtype='image'
elif problemtype=='4':
	problemtype='video'
elif problemtype=='5':
	problemtype=='csv'

print('\n OK cool, we got you modeling %s files \n'%(problemtype))
count=0
availableclasses=list()
for i in range(len(folders)):
    if data[folders[i]]==problemtype:
    	availableclasses.append(folders[i])
    	count=count+1
classnum=input('how many classes would you like to model? (%s available) \n'%(str(count)))
print('these are the available classes: ')
print(availableclasses)
classes=list()
stillavailable=list()
for i in range(int(classnum)):
	class_=input('what is class #%s \n'%(str(i+1)))

	while class_ not in availableclasses and class_ not in '' or class_ in classes:
		print('\n')
		print('------------------ERROR------------------')
		print('the input class does not exist (for %s files).'%(problemtype))
		print('these are the available classes: ')
		if len(stillavailable)==0:
			print(availableclasses)
		else:
			print(stillavailable)
		print('------------------------------------')
		class_=input('what is class #%s \n'%(str(i+1)))
	for j in range(len(availableclasses)):
		stillavailable=list()
		if availableclasses[j] not in classes:
			stillavailable.append(availableclasses[j])
	if class_ == '':
		class_=stillavailable[0]

	classes.append(class_)

mtype=input('is this a classification (c) or regression (r) problem? \n')
while mtype not in ['c','r']:
	print('input not recognized...')
	mtype=input('is this a classification (c) or regression (r) problem? \n')

# load all default feature types 
settings=json.load(open(prevdir+'/settings.json'))
default_audio_features=settings['default_audio_features']
default_text_features=settings['default_text_features']
default_image_features=settings['default_image_features']
default_video_features=settings['default_video_features']
default_csv_features='n/a'

# now featurize each class (in proper folder)
data={}
for i in range(len(classes)):
	class_type=classes[i]
	if problemtype == 'audio':
		# featurize audio 
		os.chdir(prevdir+'/features/audio_features')
		default_features=default_audio_features
	elif problemtype == 'text':
		# featurize text
		os.chdir(prevdir+'/features/text_features')
		default_features=default_text_features
	elif problemtype == 'image':
		# featurize images
		os.chdir(prevdir+'/features/image_features')
		default_features=default_image_features
	elif problemtype == 'video':
		# featurize video 
		os.chdir(prevdir+'/features/video_features')
		default_features=default_video_features
	elif problemtype == '.csv':
		# featurize .CSV 
		os.chdir(prevdir+'/features/csv_features')
		default_features=default_csv_features

	os.system('python3 featurize.py %s'%(data_dir+'/'+classes[i]))
	os.chdir(data_dir+'/'+classes[i])
	# load audio features 
	listdir=os.listdir()
	feature_list=list()
	label_list=list()
	for i in range(len(listdir)):
		if listdir[i][-5:]=='.json':
			g=json.load(open(listdir[i]))
			feature_list.append(g['features'][problemtype][default_features]['features'])
			print(g['features'][problemtype][default_features]['features'])
	
	data[class_type]=feature_list

# get feature labels (for ludwig) - should be the same for all files
feature_labels=g['features'][problemtype][default_features]['labels']

# now that we have featurizations, we can load them in and train model
os.chdir(prevdir+'/training/')

model_dir=prevdir+'/models'

jsonfile=''
for i in range(len(classes)):
    if i==0:
        jsonfile=classes[i]
    else:
        jsonfile=jsonfile+'_'+classes[i]

jsonfile=jsonfile+'.json'

#try:
g=data
alldata=list()
labels=list()
lengths=list()

# check to see all classes are same length and reshape if necessary
for i in range(len(classes)):
    class_=g[classes[i]]
    lengths.append(len(class_))

lengths=np.array(lengths)
minlength=np.amin(lengths)

# now load all the classes
for i in range(len(classes)):
    class_=g[classes[i]]
    random.shuffle(class_)

    if len(class_) > minlength:
        print('%s greater than class, equalizing...'%(class_))
        class_=class_[0:minlength]

    for j in range(len(class_)):
        alldata.append(class_[i])
        labels.append(i)

os.chdir(model_dir)

alldata=np.asarray(alldata)
labels=np.asarray(labels)

## bring main imports down here to speed up things.
## only import the training scripts that are necessary.

############################################################
## 					TRAIN THE MODEL 					  ##
############################################################

default_training_script='keras'
model_compress=True

if default_training_script=='adanet':
	print('Adanet training is coming soon! Please use a different model setting for now.') 
	# import train_adanet as ta 
	# ta.train_adanet(mtype, classes, jsonfile, alldata, labels, feature_labels, problemtype, default_features)
elif default_training_script=='alphapy':
	print('Alphapy training is coming soon! Please use a different model setting for now.') 
	# import train_alphapy as talpy
	# talpy.train_alphapy(alldata, labels, mtype, jsonfile, problemtype, default_features)
elif default_training_script=='autokeras':
	print('Autokeras training is unstable! Please use a different model setting for now.') 
	# import train_autokeras as tak 
	# tak.train_autokeras(classes, alldata, labels, mtype, jsonfile, problemtype, default_features)
elif default_training_script=='autosklearn':
	print('Autosklearn training is unstable! Please use a different model setting for now.') 
	# import train_autosklearn as taskl
	# taskl.train_autosklearn(alldata, labels, mtype, jsonfile, problemtype, default_features)
elif default_training_script=='devol':
	import train_devol as td 
	modelname, modeldir=td.train_devol(classes, alldata, labels, mtype, jsonfile, problemtype, default_features)
elif default_training_script=='hypsklearn':
	import train_hypsklearn as th 
	modelname, modeldir=th.train_hypsklearn(alldata, labels, mtype, jsonfile, problemtype, default_features)
elif default_training_script=='keras':
	import train_keras as tk
	modelname, modeldir=tk.train_keras(classes, alldata, labels, mtype, jsonfile, problemtype, default_features)
elif default_training_script=='ludwig':
	import train_ludwig as tl
	modelname, modeldir=tl.train_ludwig(mtype, classes, jsonfile, alldata, labels, feature_labels, problemtype, default_features)
elif default_training_script=='plda':
	print('PLDA training is unstable! Please use a different model setting for now.') 
	# import train_pLDA as tp
	# tp.train_pLDA(alldata,labels)
elif default_training_script=='scsr':
	import train_scsr as scsr
	if mtype == 'c':
		modelname, modeldir=scsr.train_sc(alldata,labels,mtype,jsonfile,problemtype,default_features, classes, minlength)
	elif mtype == 'r':
		modelname, modeldir=scsr.train_sr(classes, problemtype, default_features, model_dir, alldata, labels)
elif default_training_script=='tpot':
	import train_TPOT as tt
	modelname, modeldir=tt.train_TPOT(alldata,labels,mtype,jsonfile,problemtype,default_features)

############################################################
## 					COMPRESS MODELS 					  ##
############################################################

# go to model directory 
os.chdir(modeldir)

if model_compress == True:
	# now compress the model according to model type 
	if default_training_script in ['hypsklearn', 'scsr', 'tpot']:
		# all .pickle files and can compress via scikit-small-ensemble
		from sklearn.externals import joblib

		# open up model 
		loadmodel=open(modelname, 'rb')
		model = pickle.load(loadmodel)
		loadmodel.close()

		# compress - from 0 to 9. Higher value means more compression, but also slower read and write times. 
		# Using a value of 3 is often a good compromise.
		joblib.dump(model, modelname[0:-7]+'_compressed.joblib',compress=3)

		# can now load compressed models as such
		# thenewmodel=joblib.load(modelname[0:-7]+'_compressed.joblib')
		# leads to up to 10x reduction in model size and .72 sec - 0.23 secoon (3-4x faster loading model)
		# note may note work in sklearn and python versions are different from saving and loading environments. 

	elif default_training_script in ['devol', 'keras']: 
		# can compress with keras_compressor 
		import logging
		from keras.models import load_model
		from keras_compressor.compressor import compress

		logging.basicConfig(
		    level=logging.INFO,
		)

		model = load_model(modelname)
		model = compress(model, 7e-1)
		model.save(modelname[0:-3]+'_compressed.h5')

	else:
		# for everything else, we can compress pocketflow models in the future.
		print('We cannot currently compress %s models. We are working on this!! \n\n The model will remain uncompressed for now'%(default_training_script))