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


This is a simple unit test suite of the the default parameters 
of the repository. 

Note this unit_testing requires the settings.json
to defined in the base directory.
'''
import unittest, os, shutil, time, uuid, random, json, markovify, pickle
import numpy as np
import pandas as pd 

###############################################################
##                  HELPER FUNCTIONS      

##	Below are some helper functions to reduce code redundancy
## 	During the unit testing process.
###############################################################

def prev_dir(directory):
	'''
	take in a directory and get the next innermost directory 
	in the tree structure.
	
	For example, 

	directory = /Users/jim/desktop
	prev_dir(directory) --> /Users/jim
	'''
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

def clean_file(directory, clean_dir, cur_dir, train_dir, file):
	'''
	take in a few directories and output a clean file for audio,
	text, image, and video files.

	test_audio.wav --> test_audio.wav (clean)
	'''
	os.chdir(train_dir)
	try:
		os.mkdir(directory)
	except:
		shutil.rmtree(directory)
		os.mkdir(directory)

	os.chdir(directory)
	shutil.copy(cur_dir+'/'+file, train_dir+'/'+directory+'/'+file)

	os.chdir(clean_dir+'/'+directory)
	os.system('python3 clean.py %s'%(train_dir+'/'+directory))
	os.chdir(train_dir+'/'+directory)
	listdir=os.listdir()
	b=False 
	if len(listdir) == 1:
		b=True
	# remove temp directory 
	os.chdir(train_dir)
	shutil.rmtree(train_dir+'/'+directory)
	msg='failed cleaning process, file does not exist in directory'

	return b, msg

def augment_file(directory, augment_dir, cur_dir, train_dir, file):
	'''
	take in a few directories and output augmented files for audio,
	text, image, and video files.

	test_audio.wav --> test_audio.wav + tsaug_test_audio.wav

	Typically augmentation strategies add 2x more data to the original
	dataset.
	'''
	os.chdir(train_dir)
	try:
		os.mkdir(directory)
	except:
		shutil.rmtree(directory)
		os.mkdir(directory)
	shutil.copy(cur_dir+'/'+file, train_dir+'/'+directory+'/'+file)

	os.chdir(augment_dir+'/'+directory)
	os.system('python3 augment.py %s'%(train_dir+'/'+directory))
	os.chdir(train_dir+'/'+directory)

	# remove temp directory 
	listdir=os.listdir()
	b=False 
	if len(listdir) > 1:
		b=True
	os.chdir(train_dir)
	shutil.rmtree(train_dir+'/'+directory)
	msg='failed augmentation, only one file exists in the directory'
	return b, msg


def featurize_file(features_dir, cur_dir, train_dir, file, sampletype, default_features):
	'''
	take in a file and output a featurized .JSON file using
	Allie internal Feature API. 

	test.wav --> test.json, test.wav with features in test.json
	'''
	directory='%s_features'%(sampletype)
	folder=train_dir+'/'+directory
	os.chdir(train_dir)
	try:
		os.mkdir(directory)
	except:
		shutil.rmtree(directory)
		os.mkdir(directory)
	# put test audio file in directory 
	shutil.copy(cur_dir+'/'+file, folder+'/'+file)
	os.chdir(features_dir+'/%s_features/'%(sampletype))
	features_list=default_features

	for i in range(len(features_list)):
		print('------------------------------')
		print('FEATURIZING - %s'%(features_list[i].upper()))
		print('------------------------------')
		os.system('python3 featurize.py %s %s'%(folder, features_list[i]))

	# now that we have the folder let's check if the array has all the features
	os.chdir(folder)
	gopen=open('test_%s.json'%(sampletype),'r')
	g=json.load(gopen)
	features=g['features'][sampletype]
	gopen.close()
	test_features=list(features)
	if test_features == features_list:
		b=True
	else:
		b=False 
	notcount=list()
	for i in range(len(features_list)):
		if features_list[i] not in test_features:
			notcount.append(features_list[i])
	msg=str(notcount) + ' failed during featurization'
	return b, msg

def transcribe_file(train_dir, file, sampletype, default_transcript):

	os.chdir(train_dir)
	directory='%s_transcription'%(sampletype)
	folder=train_dir+'/'+directory
	os.chdir(train_dir)
	try:
		os.mkdir(directory)
	except:
		shutil.rmtree(directory)
		os.mkdir(directory)

	# put test audio file in directory 
	shutil.copy(cur_dir+'/'+file, folder+'/'+file)

	os.chdir(features_dir+'/%s_features/'%(sampletype))
	os.system('python3 featurize.py %s'%(folder))

	# now that we have the folder let's check if the array has all the features
	os.chdir(folder)
	g=json.load(open('test_%s.json'%(sampletype)))
	transcripts=list(g['transcripts'][sampletype])

	if default_transcript[0] in transcripts:
		msg='success'
		b=True
	else:
		msg='failure'
		b=False 
	os.chdir(train_dir)
	shutil.rmtree(directory)
	return b, msg

def model_predict(filetype, testfile, loadmodel_dir, load_dir):

	# copy machine learning model into image_model dir 
	os.chdir(cur_dir+'/helpers/models/%s_models/'%(filetype))
	listdir=os.listdir()
	temp=os.getcwd()
	tempfiles=list()

	os.chdir(loadmodel_dir)
	if '%s_models'%(filetype) not in os.listdir():
		os.mkdir('%s_models'%(filetype))

	os.chdir(temp)

	# copy audio machine learning model into directory (one_two)
	tempfiles=list()
	for i in range(len(listdir)):
		try:
			shutil.copytree(temp+'/'+listdir[i], loadmodel_dir+'/%s_models/'%(filetype)+listdir[i])
		except:
			pass
		tempfiles.append(listdir[i])

	# copy file in load_dir 
	shutil.copy(cur_dir+'/'+testfile, load_dir+'/'+testfile)
	# copy machine learning models into proper models directory 
	os.chdir(cur_dir+'/helpers/models/%s_models/'%(filetype))
	listdir=os.listdir()

	# copy audio machine learning model into directory (one_two)
	tempfiles=list()
	for i in range(len(listdir)):
		try:
			shutil.copytree(temp+'/'+listdir[i], loadmodel_dir+'/%s_models/'%(filetype)+listdir[i])
		except:
			pass
		tempfiles.append(listdir[i])

	os.chdir(loadmodel_dir)
	os.system('python3 load.py')
	os.chdir(load_dir)

	os.chdir(load_dir)
	listdir=os.listdir() 
	b=False
	for i in range(len(listdir)):
		if filetype == 'audio':
			if listdir[i].endswith('.wav') and listdir[i][0:-4]+'.json' in listdir: 
				b=True
				break
		elif filetype == 'text':
			if listdir[i].endswith('.txt') and listdir[i][0:-4]+'.json' in listdir: 
				b=True
				break
		elif filetype == 'image':
			if listdir[i].endswith('.png') and listdir[i][0:-4]+'.json' in listdir: 
				b=True
				break
		elif filetype == 'video':
			if listdir[i].endswith('.mp4') and listdir[i][0:-4]+'.json' in listdir: 
				b=True
				break

	# now remove all the temp files 
	os.chdir(loadmodel_dir+'/%s_models'%(filetype))
	for i in range(len(tempfiles)):
		shutil.rmtree(tempfiles[i])

	msg = filetype + ' model prediction failed.'
	return b, msg

###############################################################
##                     INITIALIZATION                        ##
###############################################################

# initialize variables for the test 
cur_dir=os.getcwd()
prevdir= prev_dir(cur_dir)
load_dir = prevdir+'/load_dir'
train_dir = prevdir + '/train_dir'
model_dir = prevdir+ '/training'
features_dir=prevdir+'/features'
loadmodel_dir = prevdir+'/models'
clean_dir=prevdir+'/cleaning/'
augment_dir=prevdir+'/augmentation'
test_dir=prevdir+'/tests'
visualization_dir=prevdir+'/visualize'
preprocessing_dir=prevdir+'/preprocessing'

# settings
settings=json.load(open(prevdir+'/settings.json'))
clean_data=settings['clean_data']
augment_data=settings['augment_data']

# transcript settings 
default_audio_transcript=settings['default_audio_transcriber']
default_image_transcript=settings['default_image_transcriber']
default_text_transcript=settings['default_text_transcriber']
default_video_transcript=settings['default_video_transcriber']
default_csv_transcript=settings['default_csv_transcriber']
transcribe_audio=settings['transcribe_audio']
transcribe_text=settings['transcribe_text']
transcribe_image=settings['transcribe_image']
transcribe_video=settings['transcribe_video'] 
transcribe_csv=settings['transcribe_csv']

# feature settings 
default_audio_features=settings['default_audio_features']
default_text_features=settings['default_text_features']
default_image_features=settings['default_image_features']
default_video_features=settings['default_video_features']
default_csv_features=settings['default_csv_features']

# cleaning settings
default_audio_cleaners=settings['default_audio_cleaners']
default_text_cleaners=settings['default_text_cleaners']
default_image_cleaners=settings['default_image_cleaners']
default_video_cleaners=settings['default_video_cleaners']
default_csv_cleaners=settings['default_csv_cleaners']

# augmentation settings 
default_audio_augmenters=settings['default_audio_augmenters']
default_text_augmenters=settings['default_text_augmenters']
default_image_augmenters=settings['default_image_augmenters']
default_video_augmenters=settings['default_video_augmenters']
default_csv_augmenters=settings['default_csv_augmenters']

# preprocessing settings
select_features=settings['select_features']
reduce_dimensions=settings['reduce_dimensions']
scale_features=settings['scale_features']
default_scaler=settings['default_scaler']
default_feature_selector=settings['default_feature_selector']
default_dimensionality_reducer=settings['default_dimensionality_reducer']
dimension_number=settings['dimension_number']
feature_number=settings['feature_number']

# other settings for raining scripts 
training_scripts=settings['default_training_script']
model_compress=settings['model_compress']

# directories 
audiodir=loadmodel_dir+'/audio_models'
textdir=loadmodel_dir+'/text_models'
imagedir=loadmodel_dir+'/image_models'
videodir=loadmodel_dir+'/video_models'
csvdir=loadmodel_dir+'/csv_models'

###############################################################
##                        UNIT TESTS                         ##
###############################################################

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_dependencies(unittest.TestCase):
	'''
	DEPENDENCY TESTS

	Confirms that all the modules are installed correctly, along with
	all brew install commands. 
	'''
#### ##### ##### ##### ##### ##### ##### ##### ##### #####
	
	def test_sox(self):
		# test brew installation by merging two test files 
		os.chdir(cur_dir)
		os.system('sox test_audio.wav test_audio.wav test2.wav')
		if 'test2.wav' in os.listdir():
			b=True  
		else:
			b=False    
		self.assertEqual(True, b)      

	
	def test_c_ffmpeg(self):
		# test FFmpeg installation with test_audio file conversion 
		os.chdir(cur_dir)
		if 'test_audio.mp3' in os.listdir():
			os.remove('test_audio.mp3')
		os.system('ffmpeg -i test_audio.wav test_audio.mp3 -y')
		if 'test_audio.mp3' in os.listdir():
			b=True 
		else:
			b=False 
		self.assertEqual(True, b)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_cleaning(unittest.TestCase):
	'''
	CLEANING API TESTS

	Tests file cleaning capabilities by removing duplicates, etc.
	across all file types.
	'''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

	def test_audio_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir):
		directory='audio_cleaning'
		file='test_audio.wav'

		b, msg = clean_file(directory, clean_dir, cur_dir, train_dir, file)

		self.assertEqual(True, b) 

	def test_text_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir):
		directory='text_cleaning'
		file='test_text.txt'

		b, msg = clean_file(directory, clean_dir, cur_dir, train_dir, file)

		self.assertEqual(True, b) 

	def test_image_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir):
		directory='image_cleaning'
		file-'test_image.png'

		b, msg = clean_file(directory, clean_dir, cur_dir, train_dir, file)

		self.assertEqual(True, b) 

	def test_video_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir):
		directory='video_cleaning'
		file='test_video.mp4'

		b, msg = clean_file(directory, clean_dir, cur_dir, train_dir, file)

		self.assertEqual(True, b) 

	def test_csv_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir):
		directory='csv_cleaning'
		file='test_csv.csv'

		b, msg = clean_file(directory, clean_dir, cur_dir, train_dir, file)

		self.assertEqual(True, b) 

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_augmentation(unittest.TestCase):
	'''
	AUGMENTATION API TESTS

	Tests augmentation capabilities for all data types.
	'''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

	def test_audio_augment(self, augment_dir=augment_dir, train_dir=train_dir, cur_dir=cur_dir):
		directory='audio_augmentation'
		file='test_audio.wav'

		b, msg = augment_file(directory, augment_dir, cur_dir, train_dir, file)

		self.assertEqual(True, b) 
		
	def test_text_augment(self, augment_dir=augment_dir, train_dir=train_dir, cur_dir=cur_dir):
		directory='text_augmentation'
		file='test_text.wav'

		b, msg = augment_file(directory, augment_dir, cur_dir, train_dir, file)

		self.assertEqual(True, b) 

	def test_image_augment(self, augment_dir=augment_dir, train_dir=train_dir, cur_dir=cur_dir):
		directory='image_augmentation'
		file='test_image.png'

		b, msg = augment_file(directory, augment_dir, cur_dir, train_dir, file)

		self.assertEqual(True, b) 

	def test_video_augment(self, augment_dir=augment_dir, train_dir=train_dir, cur_dir=cur_dir):
		directory='video_augmentation'
		file='test_video.mp4'

		b, msg=augment_file(directory, augment_dir, cur_dir, train_dir, file)

		self.assertEqual(True, b) 

	def test_csv_augment(self, augment_dir=augment_dir, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir):
		directory='csv_augmentation'
		file='test_csv.csv'

		b, msg = augment_file(directory, augment_dir, cur_dir, train_dir, file)

		self.assertEqual(True, b) 

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_features(unittest.TestCase):
	'''
	FEATURIZATION API TESTS

	Tests featurization capabilities across all training scripts.
	'''
#### ##### ##### ##### ##### ##### ##### ##### ##### #####

	def test_audio_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir, default_audio_features=default_audio_features):
		file='test_audio.wav'
		sampletype='audio'
		default_features=default_audio_features

		b, msg = featurize_file(features_dir, cur_dir, train_dir, file, sampletype, default_features)
		
		self.assertEqual(True, b, msg) 
		
	def test_text_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir, default_text_features=default_text_features):
		file='test_text.txt'
		sampletype='text'
		default_features=default_text_features

		b, msg = featurize_file(features_dir, cur_dir, train_dir, file, sampletype, default_features)
		
		self.assertEqual(True, b, msg) 

	def test_image_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir, default_image_features=default_image_features):
		file='test_image.png'
		sampletype='image'
		default_features=default_image_features

		b, msg = featurize_file(features_dir, cur_dir, train_dir, file, sampletype, default_features)
		
		self.assertEqual(True, b, msg) 

	def test_video_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir, default_video_features=default_video_features):
		file='test_video.mp4'
		sampletype='video'
		default_features=default_video_features

		b, msg = featurize_file(features_dir, cur_dir, train_dir, file, sampletype, default_features)
		
		self.assertEqual(True, b, msg) 

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_transcription(unittest.TestCase):
	'''
	TRANSCRIPTION API TESTS

	tests the ability to transcribe across many
	data types
	'''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

	def setUp(self, prevdir=prevdir):
		# change settings.json to test all model scripts
		os.system('pip3 install opencv-python==3.4.2.16 opencv-contrib-python==3.4.2.16') 
		os.chdir(prevdir)
		settings=json.load(open('settings.json'))
		settings['transcribe_audio']=True
		settings['transcribe_text']=True
		settings['transcribe_image']=True
		settings['transcribe_videos']=True 
		settings['transcribe_csv']=True
		jsonfile=open('settings.json', 'w')
		json.dump(settings, jsonfile)
		jsonfile.close()

	def tearDown(self, prevdir=prevdir, transcribe_audio=transcribe_audio, transcribe_text=transcribe_text, transcribe_image=transcribe_image, transcribe_video=transcribe_video, transcribe_csv=transcribe_csv):
		# change settings.json back to normal to defaults  
		os.chdir(prevdir)
		settings=json.load(open('settings.json'))
		settings['transcribe_audio']=transcribe_audio
		settings['transcribe_text']=transcribe_text
		settings['transcribe_image']=transcribe_image
		settings['transcribe_videos']=transcribe_video
		settings['transcribe_csv']=transcribe_csv
		jsonfile=open('settings.json','w')
		json.dump(settings, jsonfile)
		jsonfile.close() 

	# audio transcription
	def test_audio_transcription(self, default_audio_transcript=default_audio_transcript, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir):
		file='test_audio.wav'
		sampletype='audio'
		default_transcript=default_audio_transcript

		b, msg = transcribe_file(train_dir, file, sampletype, default_transcript)

		self.assertEqual(True, b, msg) 

	# text transcription
	def test_text_transcription(self, default_text_transcript=default_text_transcript, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir):
		file='test_text.txt'
		sampletype='text'
		default_transcript=default_text_transcript

		b, msg = transcribe_file(train_dir, file, sampletype, default_transcript)

		self.assertEqual(True, b, msg) 

	# image transcription
	def test_image_transcription(self, default_image_transcript=default_image_transcript, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir):
		file='test_image.png'
		sampletype='image'
		default_transcript=default_image_transcript

		b, msg = transcribe_file(train_dir, file, sampletype, default_transcript)

		self.assertEqual(True, b, msg) 

	# video transcription
	def test_video_transcription(self, default_video_transcript=default_video_transcript, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir):
		file='test_video.mp4'
		sampletype='video'
		default_transcript=default_video_transcript

		b, msg = transcribe_file(train_dir, file, sampletype, default_transcript)

		self.assertEqual(True, b, msg) 

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_training(unittest.TestCase):
	'''
	MODEL TRAINING API TESTS

	Tests all available training scripts and compression abilities.
	'''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

	def setUp(self, prevdir=prevdir, training_scripts=training_scripts):
		# change settings.json to test all model scripts 
		os.chdir(prevdir)
		gopen=open('settings.json','r')
		settings=json.load(gopen)
		gopen.close()
		settings['default_training_script']=training_scripts
		settings["default_text_features"] = ["nltk_features"]
		settings['select_features']=False
		settings['scale_features']=False
		settings['reduce_dimensions']=False
		settings['remove_outliers']=True
		settings['visualize_data']=False
		settings['clean_data']=False 
		settings['augment_data']=False 
		settings['model_compress']=False 
		jsonfile=open('settings.json', 'w')
		json.dump(settings, jsonfile)
		jsonfile.close()

	def tearDown(self, textdir=textdir, prevdir=prevdir, training_scripts=training_scripts, clean_data=clean_data, augment_data=augment_data, model_compress=model_compress):
		# change settings.json back to normal to defaults  
		os.chdir(prevdir)
		gopen=open('settings.json','r')
		settings=json.load(gopen)
		gopen.close()
		settings['default_training_script']=training_scripts
		settings['clean_data']=clean_data
		settings['augment_data']=augment_data 
		settings['model_compress']=model_compress
		jsonfile=open('settings.json','w')
		json.dump(settings, jsonfile)
		jsonfile.close() 

	def test_training(self, cur_dir=cur_dir, train_dir=train_dir, model_dir=model_dir, clean_data=clean_data, augment_data=augment_data, test_dir=test_dir):
		# use text model for training arbitrarily because it's the fastest model training time.
		# note that the files here are already featurized to only test modeling capability (and not featurization or other aspects of the Models API)
		os.chdir(train_dir)
		shutil.copytree(test_dir+'/helpers/model_test/one', os.getcwd()+'/one')
		shutil.copytree(test_dir+'/helpers/model_test/two', os.getcwd()+'/two')	

		os.chdir(model_dir)
		# iterate through all machine learning model training methods
		os.system('python3 model.py text 2 c onetwo one two')
		os.chdir(train_dir)
		shutil.rmtree('one')
		shutil.rmtree('two')
		
		# now find the model 
		os.chdir(textdir)
		listdir=os.listdir()
		b=False

		# remove temporary files in the textdir
		for i in range(len(listdir)):
			if listdir[i].find('onetwo') >= 0:
				b=True
				# use shutil to remove a folder. 
				shutil.rmtree(listdir[i])
				break
			else:
				os.remove(listdir[i])

		self.assertEqual(True, b)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_loading(unittest.TestCase):
	'''
	LOADING API TESTS

	Note we have to do a loop here to end where the end is 
	'audio.json' | 'text.json' | 'image.json' | 'video.json' | 'csv.json'
	this is because the files are renamed to not have conflicts.
	
	For example, if 'audio.wav' --> 'audio.json' and 'audio.mp4' --> 'audio.json',
	both would have a conflicting name and would overwrite each other. 
	'''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

	def setUp(self, prevdir=prevdir):
		# change settings.json to test all model scripts 
		os.chdir(prevdir)
		gopen=open('settings.json','r')
		settings=json.load(gopen)
		gopen.close()
		# set features for the right ML models 
		settings['default_audio_features']=['librosa_features'] 
		settings['default_text_features']=['nltk_features'] 
		settings['default_image_features']=['image_features'] 
		settings['default_video_features']=['video_features'] 
		jsonfile=open('settings.json', 'w')
		json.dump(settings, jsonfile)
		jsonfile.close()

	def tearDown(self, default_audio_features=default_audio_features, default_text_features=default_text_features, default_image_features=default_image_features, default_video_features=default_video_features, default_csv_features=default_csv_features):
		os.chdir(prevdir)
		gopen=open('settings.json','r')
		settings=json.load(gopen)
		gopen.close()
		# set features back to what they were before. 
		settings['default_audio_features']=default_audio_features
		settings['default_text_features']=default_text_features
		settings['default_image_features']=default_image_features
		settings['default_video_features']=default_video_features 
		settings['default_csv_features']=default_csv_features 
		jsonfile=open('settings.json','w')
		json.dump(settings, jsonfile)
		jsonfile.close() 

	def test_loadaudio(self, load_dir=load_dir, cur_dir=cur_dir, loadmodel_dir=loadmodel_dir):
		filetype='audio'
		testfile='test_audio.wav'

		b, msg = model_predict(filetype, testfile, loadmodel_dir, load_dir)

		self.assertEqual(True, b)

	def test_loadtext(self, load_dir=load_dir, cur_dir=cur_dir, loadmodel_dir=loadmodel_dir):
		filetype='text'
		testfile='test_text.txt'

		b, msg = model_predict(filetype, testfile, loadmodel_dir, load_dir)

		self.assertEqual(True, b)

	def test_loadimage(self, load_dir=load_dir, cur_dir=cur_dir, loadmodel_dir=loadmodel_dir):
		filetype='image'
		testfile='test_image.png'

		b, msg = model_predict(filetype, testfile, loadmodel_dir, load_dir)

		self.assertEqual(True, b)

	def test_loadvideo(self, load_dir=load_dir, cur_dir=cur_dir, loadmodel_dir=loadmodel_dir):
		filetype='video'
		testfile='test_video.mp4'

		b, msg = model_predict(filetype, testfile, loadmodel_dir, load_dir)

		self.assertEqual(True, b)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_preprocessing(unittest.TestCase):
	'''
	PREPROCESSING API TESTS

	Tests Allie's preprocessing functionality to reduce dimensionality, 
	select features, and scale features.
	'''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

	def setUp(self, prevdir=prevdir):
		# change settings.json to test all model scripts 
		os.chdir(prevdir)
		gopen=open('settings.json','r')
		settings=json.load(gopen)
		gopen.close()
		# set features for the right ML models 
		settings['select_features']=True
		settings['reduce_dimensions']=True
		settings['scale_features']=True 
		settings['default_scaler']=["standard_scaler"]
		settings['default_feature_selector']=["rfe"]
		settings['default_dimensionionality_reducer']=["pca"]
		settings['dimension_number']=20
		settings['feature_number']=2
		jsonfile=open('settings.json', 'w')
		json.dump(settings, jsonfile)
		jsonfile.close()

	def tearDown(self, prevdir=prevdir, select_features=select_features, reduce_dimensions=reduce_dimensions, scale_features=scale_features, default_scaler=default_scaler, default_feature_selector=default_feature_selector, default_dimensionality_reducer=default_dimensionality_reducer, dimension_number=dimension_number, feature_number=feature_number):
		os.chdir(prevdir)
		gopen=open('settings.json','r')
		settings=json.load(gopen)
		gopen.close()

		# set features for the right ML models 
		settings['select_features']=select_features
		settings['reduce_dimensions']=reduce_dimensions
		settings['scale_features']=scale_features
		settings['default_scaler']=default_scaler
		settings['default_feature_selector']=default_feature_selector
		settings['default_dimensionality_reducer']=default_dimensionality_reducer
		settings['dimension_number']=dimension_number
		settings['feature_number']=feature_number

		jsonfile=open('settings.json','w')
		json.dump(settings, jsonfile)
		jsonfile.close() 

	def test_createtransformer(self, preprocessing_dir=preprocessing_dir, test_dir=test_dir):
		# copy files into the train_dir
		os.chdir(test_dir)
		try:
			shutil.copytree(test_dir+'/helpers/model_test/one', train_dir+'/one')
		except:
			shutil.rmtree(train_dir+'/one')
			shutil.copytree(test_dir+'/helpers/model_test/one', train_dir+'/one')
		try:
			shutil.copytree(test_dir+'/helpers/model_test/two', train_dir+'/two')
		except:
			shutil.rmtree(train_dir+'/two')
			shutil.copytree(test_dir+'/helpers/model_test/two', train_dir+'/two')

		os.chdir(preprocessing_dir)
		# call it using proper format 
		os.system('python3 transform.py text c onetwo one two')
		# now that we have transformer test to see if it exists 
		if 'text_transformer' in os.listdir():
			os.chdir('text_transformer')
			listdir=os.listdir()
			if 'c_onetwo_standard_scaler_pca_rfe.json' in listdir:
				b=True
			else:
				b=False
		else:
			b=False

		shutil.rmtree(train_dir+'/one')
		shutil.rmtree(train_dir+'/two')

		# feature select data 
		self.assertEqual(True, b)

	def test_loadtransformer(self, test_dir=test_dir, preprocessing_dir=preprocessing_dir): 
		try:
			shutil.copytree(test_dir+'/helpers/text_transformer', preprocessing_dir+'/text_transformer/')
		except:
			shutil.rmtree(preprocessing_dir+'/text_transformer/')
			shutil.copytree(test_dir+'/helpers/text_transformer', preprocessing_dir+'/text_transformer/')

		# now actually convert and load data with this transformer 
		os.chdir(preprocessing_dir+'/text_transformer/')
		model=pickle.load(open('c_onetwo_standard_scaler_pca_rfe.pickle','rb'))
		jsonfile=json.load(open('c_onetwo_standard_scaler_pca_rfe.json'))
		sample=jsonfile['sample input X']
		transformed_sample=jsonfile['sample transformed X']

		newsize=model.transform(np.array(sample).reshape(1,-1))

		# ---> FOR TESTING ONLY <----
		# print(model)
		# print(newsize)
		# print(type(newsize))
		# print(transformed_sample)
		# print(type(transformed_sample))

		if np.size(newsize[0]) == np.size(np.array(transformed_sample)):
			b=True
		else:
			b=False

		self.assertEqual(True, b)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_visualization(unittest.TestCase):
	'''
	VISUALIZATION API TESTS

	Tests Allie's visualization API capabilities for classification problems.
	'''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

	def test_visualization(self, test_dir=test_dir, train_dir=train_dir, visualization_dir=visualization_dir):
		# copy files into the train_dir
		os.chdir(test_dir)
		shutil.copytree(test_dir+'/helpers/model_test/one', train_dir+'/one')
		shutil.copytree(test_dir+'/helpers/model_test/two', train_dir+'/two')

		# now run the visualization
		os.chdir(visualization_dir)
		if 'visualization_session' in os.listdir():
			shutil.rmtree('visualization_session')
		os.system('python3 visualize.py text one two')
		if 'visualization_session' in os.listdir():
			os.chdir('visualization_session')
			files=os.listdir()
			if 'clustering' in files and 'feature_ranking' in files and 'model_selection' in files and 'classes.png' in files:
				b=True
			else:
				b=False
		else:
			b=False

		os.chdir(train_dir)
		shutil.rmtree("one")		
		shutil.rmtree("two")

		# visualize data (text data featurized)
		self.assertEqual(True, b)

if __name__ == '__main__':
	unittest.main()