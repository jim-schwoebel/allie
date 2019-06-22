'''
Take in various sys.argv[] inputs and/or user-defined inputs and
output a trained machine learning model. 

Can featurize and model audio, image, text, video, or CSV files.

This assumes the files are separated in folders in the train_dir. 

For example, if you want to separate males and females in audio
files, you would need to have one folder named 'males' and another
folder named 'females' and specify audio file training with the
proper folder names ('male' and 'female') for the training script 
to function properly.

For automated training, you can alternatively pass through sys.argv[]
inputs as follows:

python3 model.py audio 2 c male female 

audio = audio file type 
2 = 2 classes 
c = classification (r for regression)
male = first class
female = second class [via N number of classes]
'''
###############################################################
##                  IMPORT STATEMENTS                        ##
###############################################################
import os, sys, pickle, json, random, shutil, time
import numpy as np
import matplotlib.pyplot as plt

###############################################################
##                  HELPER FUNCTIONS                         ##
###############################################################

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

###############################################################
##                    LOADING SETTINGS                       ##
###############################################################

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
default_csv_features=settings['default_csv_features']

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

###############################################################
##                  INITIALIZE CLASSES                       ##
###############################################################

# get all information from sys.argv, and if not, 
# go through asking user for the proper parameters 

try:
	problemtype=sys.argv[1]
	classnum=sys.argv[2]
	mtype=sys.argv[3]
	classes=list()
	for i in range(int(classnum)):
		classes.append(sys.argv[i+4])
except:
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

	common_name=input('what is the 1-word common name for the problem you are working on? (e.g. gender for male/female classification) \n')
	mtype=input('is this a classification (c) or regression (r) problem? \n')
	while mtype not in ['c','r']:
		print('input not recognized...')
		mtype=input('is this a classification (c) or regression (r) problem? \n')

# print(problemtype)
# time.sleep(10)
###############################################################
##                    FEATURIZE FILES                        ##
###############################################################

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
	elif problemtype == 'csv':
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
			# consolidate all features into one array (if featurizing with multiple featurizers)
			default_feature=list()
			for j in range(len(default_features)):
				default_feature=default_feature+g['features'][problemtype][default_features[j]]['features']

			feature_list.append(default_feature)
			print(default_feature)
	
	data[class_type]=feature_list

# get feature labels (for ludwig) - should be the same for all files
feature_labels=g['features'][problemtype][default_features[0]]['labels']

###############################################################
##                    DATA PRE-PROCESSING                    ##
###############################################################

# perform class balance such that both classes have the same number
# of members. 

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

############################################################
## 					TRAIN THE MODEL 					  ##
############################################################

## Now we can train the machine learning model via the default_training script.

default_training_script=settings['default_training_script']
model_compress=settings['model_compress']

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


############################################################
## 					CREATE YAML FILES 					  ##
############################################################

create_YAML=settings['create_YAML']

# note this will only work for pickle files for now. Will upgrade to deep learning
# models in the future.
if create_YAML == True and default_training_script in ['scsr', 'tpot', 'hypsklearn']:

	# clear the load directory 
	os.chdir(prevdir+'/load_dir/')
	listdir=os.listdir()
	for i in range(len(listdir)):
		os.remove(listdir[i])

	# First need to create some test files for each use case
	# copy 10 files from each class into the load_dir 
	copylist=list()
	print(classes)

	# clear the load_directory 
	os.chdir(prevdir+'/load_dir/')
	listdir2=os.listdir()
	for j in range(len(listdir2)):
		# remove all files in the load_dir 
		os.remove(listdir2[j])	

	for i in range(len(classes)):
		os.chdir(prevdir+'/train_dir/'+classes[i])
		tempdirectory=os.getcwd()
		listdir=os.listdir()
		os.chdir(tempdirectory)
		count=0
		
		# rename files if necessary 
		# the only other file in the directory should be .json, so this should work
		for j in range(len(listdir)):
			# print(listdir[j])
			if listdir[j].replace(' ','_').replace('-','') not in copylist and listdir[j].endswith('.json') == False:
				if count >= 10:
					break
				else:
					newname=listdir[j].replace(' ','_').replace('-','')
					os.rename(listdir[j], newname)
					shutil.copy(tempdirectory+'/'+newname, prevdir+'/load_dir/'+newname)
					# print(tempdirectory+'/'+listdir[j])
					copylist.append(newname)
					count=count+1 

			elif listdir[j].replace(' ','_').replace('-','') in copylist and listdir[j].endswith('.json') == False: 
				# rename the file appropriately such that there are no conflicts.
				if count >= 10:
					break
				else:
					newname=str(j)+'_'+listdir[j].replace(' ','_').replace('-','')
					os.rename(listdir[j], newname)
					# print(tempdirectory+'/'+newname)
					shutil.copy(tempdirectory+'/'+newname, prevdir+'/load_dir/'+newname)
					copylist.append(newname)
					count=count+1 

	# now apply machine learning model to these files 
	os.chdir(prevdir+'/models')
	os.system('python3 load_models.py')

	# now load up each of these and break loop when tests for each class exist 
	os.chdir(prevdir+'/load_dir')
	jsonfiles=list()
	# iterate through until you find a class that fits in models 
	listdir=os.listdir()

	for i in range(len(classes)):
		for j in range(len(listdir)):
			if listdir[j][-5:]=='.json':
				try:
					g=json.load(open(listdir[j]))
					models=g['models'][problemtype][classes[i]]
					print(models)
					features=list()
					for k in range(len(default_features)):
						features=features+g['features'][problemtype][default_features[k]]['features']
					
					print(features)
					testfile=classes[i]+'_features.json'
					print(testfile)
					jsonfile=open(testfile,'w')
					json.dump(features,jsonfile)
					jsonfile.close()
					shutil.move(os.getcwd()+'/'+testfile, prevdir+'/models/'+problemtype+'_models/'+testfile)
					break
				except:
					pass 

	# now create the YAML file 
	os.chdir(prevdir+'/production')
	cmdclasses=''
	for i in range(len(classes)):
		if i != len(classes)-1:
			cmdclasses=cmdclasses+classes[i]+' '
		else:
			cmdclasses=cmdclasses+classes[i]

	if 'common_name' in locals():
		pass 
	else:
		common_name=modelname[0:-7]

	jsonfilename=modelname[0:-7]+'.json'
	os.system('python3 create_yaml.py %s %s %s %s %s'%(problemtype, common_name, modelname, jsonfilename, cmdclasses))


