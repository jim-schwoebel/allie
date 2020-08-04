'''
Simple unit test the default parameters 
of the repository. 

Note this unit_testing requires the settings.json
to defined in the base directory.
'''
import unittest, os, shutil, time, uuid, random, json, markovify, pickle
import numpy as np
import pandas as pd 

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

def seed_files(filename, cur_dir, train_dir, clean_data, augment_data, filetype):
	os.chdir(train_dir)

	def text_record(filename, text_model):
		textfile=open(filename, 'w')
		# Print five randomly-generated sentences
		for i in range(5):
			textfile.write(text_model.make_sentence())
		textfile.close()

	# Get raw text as string (the Brother's Karamazov, one of my fav novels)
	with open(cur_dir+'/helpers/text.txt') as f:
		text = f.read()
	# Build the model.
	text_model = markovify.Text(text)
	for i in range(20):
		filename=str(uuid.uuid4())+'.txt'
		text_record(filename, text_model)

def find_model(b):
	listdir=os.listdir()
	b=0
	for i in range(len(listdir)):
		if listdir[i].find('one_two') == 0 and listdir[i].endswith('.h5'):
			# this is for .h5 models 
			b=b+1 
		elif listdir[i].find('one_two') == 0 and listdir[i].endswith('.pickle'):
			# this is for pickle models 
			b=b+1
		elif listdir[i].find('one_two') == 0 and listdir[i].endswith('.joblib'):
			# this is for compressed joblib models 
			b=b+1
		elif listdir[i].find('one_two_ludwig_nltk_features.hdf5') == 0:
			b=b+1

	# 6=non-compressed, 12=compressed models 
	if b == 6*2:
		b=True
	else:
		b=False 

	print(b)

	return b

def clean_file(directory, clean_dir, train_dir):
	os.chdir(clean_dir)
	os.system('python3 clean.py %s %s'%(clean_dir, train_dir+'/'+directory))

def augment_file(directory, augment_dir, train_dir):
	os.chdir(augment_dir)
	os.system('python3 augment.py %s'%(train_dir+'/'+directory))

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
	confirm that all the modules are installed correctly, along with
	all brew install commands. 
	'''
#### ##### ##### ##### ##### ##### ##### ##### ##### #####
	
	# brew installations (SoX)
	def test_sox(self):

		# test brew installation by merging two test files 
		os.chdir(cur_dir)
		os.system('sox test_audio.wav test_audio.wav test2.wav')
		if 'test2.wav' in os.listdir():
			b=True  
		else:
			b=False    
		self.assertEqual(True, b)      

	# brew installation (FFmpeg)
	def test_c_ffmpeg(self):

		# test FFmpeg installation with test_audio file conversion 
		os.chdir(cur_dir)
		if 'test_audio.mp3' in os.listdir():
			# remove temp file if it already exists
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
	tests file cleaning capabilities by removing duplicates, etc.
	across all file types.
	'''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

	# audio file cleaning
	def test_audio_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
		directory='audio_cleaning'
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		os.chdir(directory)
		shutil.copy(cur_dir+'/test_audio.wav', train_dir+'/'+directory+'/test_audio.wav')

		os.chdir(clean_dir+'/'+directory)
		os.system('python3 clean.py %s'%(train_dir+'/'+directory))
		os.chdir(train_dir+'/'+directory)

		listdir=os.listdir()
		b=False 
		if len(listdir) == 1:
			b=True
		self.assertEqual(True, b) 
		# remove temp directory 
		os.chdir(train_dir)
		shutil.rmtree(train_dir+'/'+directory)

	# text file cleaning 
	def test_text_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
		directory='text_cleaning'
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		os.chdir(directory)
		shutil.copy(cur_dir+'/test_text.txt', train_dir+'/'+directory+'/test_text.txt')
		
		os.chdir(clean_dir+'/'+directory)
		os.system('python3 clean.py %s'%(train_dir+'/'+directory))
		os.chdir(train_dir+'/'+directory)

		listdir=os.listdir()
		b=False 
		if len(listdir) == 1:
			b=True
		self.assertEqual(True, b) 
		# remove temp directory 
		os.chdir(train_dir)
		shutil.rmtree(train_dir+'/'+directory)

	# image file cleaning 
	def test_image_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
		directory='image_cleaning'
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		os.chdir(directory)
		shutil.copy(cur_dir+'/test_image.png', train_dir+'/'+directory+'/test_image.png')
		
		os.chdir(clean_dir+'/'+directory)
		os.system('python3 clean.py %s'%(train_dir+'/'+directory))
		os.chdir(train_dir+'/'+directory)

		listdir=os.listdir()
		b=False 
		if len(listdir) == 1:
			b=True
		self.assertEqual(True, b) 
		# remove temp directory 
		os.chdir(train_dir)
		shutil.rmtree(train_dir+'/'+directory)

	# video file cleaning
	def test_video_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
		directory='video_cleaning'
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		os.chdir(directory)
		shutil.copy(cur_dir+'/test_video.mp4', train_dir+'/'+directory+'/test_video.mp4')

		os.chdir(clean_dir+'/'+directory)
		os.system('python3 clean.py %s'%(train_dir+'/'+directory))
		os.chdir(train_dir+'/'+directory)

		listdir=os.listdir()
		b=False 
		if len(listdir) == 1:
			b=True
		self.assertEqual(True, b) 
		# remove temp directory 
		os.chdir(train_dir)
		shutil.rmtree(train_dir+'/'+directory)

	# csv file cleaning
	def test_csv_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
		directory='csv_cleaning'
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		os.chdir(directory)
		shutil.copy(cur_dir+'/test_csv.csv', train_dir+'/'+directory+'/test_csv.csv')

		os.chdir(clean_dir+'/'+directory)
		os.system('python3 clean.py %s'%(train_dir+'/'+directory))
		os.chdir(train_dir+'/'+directory)

		os.chdir(train_dir+'/'+directory)
		listdir=os.listdir()
		b=False 
		if len(listdir) > 1:
			b=True
		self.assertEqual(True, b) 
		# remove temp directory 
		os.chdir(train_dir)
		shutil.rmtree(train_dir+'/'+directory)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_augmentation(unittest.TestCase):
	'''
	tests augmentation capabilities for all data types.
	'''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

	# audio features
	def test_audio_augment(self, augment_dir=augment_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
		directory='audio_augmentation'
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		shutil.copy(cur_dir+'/test_audio.wav', train_dir+'/'+directory+'/test_audio.wav')

		os.chdir(augment_dir+'/'+directory)
		os.system('python3 augment.py %s'%(train_dir+'/'+directory))
		os.chdir(train_dir+'/'+directory)

		listdir=os.listdir()
		b=False 
		if len(listdir) > 1:
			b=True
		print('---------STOP------------')
		print(len(listdir))
		self.assertEqual(True, b) 
		# remove temp directory 
		os.chdir(train_dir)
		shutil.rmtree(train_dir+'/'+directory)
		
	# text features
	def test_text_augment(self, augment_dir=augment_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
		directory='text_augmentation'
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)
		shutil.copy(cur_dir+'/test_text.txt', train_dir+'/'+directory+'/test_text.txt')

		os.chdir(augment_dir+'/'+directory)
		os.system('python3 augment.py %s'%(train_dir+'/'+directory))
		os.chdir(train_dir+'/'+directory)

		listdir=os.listdir()
		b=False 
		if len(listdir) > 1:
			b=True
		self.assertEqual(True, b) 
		# remove temp directory 
		os.chdir(train_dir)
		shutil.rmtree(train_dir+'/'+directory)

	# image features
	def test_image_augment(self, augment_dir=augment_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
		directory='image_augmentation'
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)
		shutil.copy(cur_dir+'/test_image.png', train_dir+'/'+directory+'/test_image.png')

		os.chdir(augment_dir+'/'+directory)
		os.system('python3 augment.py %s'%(train_dir+'/'+directory))
		os.chdir(train_dir+'/'+directory)

		listdir=os.listdir()
		b=False 
		if len(listdir) > 1:
			b=True
		self.assertEqual(True, b) 
		# remove temp directory 
		os.chdir(train_dir)
		shutil.rmtree(train_dir+'/'+directory)

	# video features
	def test_video_augment(self, augment_dir=augment_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
		directory='video_augmentation'
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)
		shutil.copy(cur_dir+'/test_video.mp4', train_dir+'/'+directory+'/test_video.mp4')

		os.chdir(augment_dir+'/'+directory)
		os.system('python3 augment.py %s'%(train_dir+'/'+directory))
		os.chdir(train_dir+'/'+directory)

		listdir=os.listdir()
		b=False 
		if len(listdir) > 1:
			b=True
		self.assertEqual(True, b) 
		# remove temp directory 
		os.chdir(train_dir)
		shutil.rmtree(train_dir+'/'+directory)

	# csv features 
	def test_csv_augment(self, augment_dir=augment_dir, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
		directory='csv_augmentation'
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		shutil.copy(cur_dir+'/test_csv.csv', train_dir+'/'+directory+'/test_csv.csv')
		
		os.chdir(clean_dir+'/csv_augmentation')
		os.system('python3 augment.py %s'%(train_dir+'/'+directory))
		os.chdir(train_dir+'/'+directory)
		os.remove('test_csv.csv')
		os.rename('augmented_combined_test_csv.csv','test_csv.csv')
		os.chdir(augment_dir+'/'+directory)
		os.system('python3 augment.py %s'%(train_dir+'/'+directory))
		os.chdir(train_dir+'/'+directory)
		
		listdir=os.listdir()
		b=False 
		if len(listdir) > 1:
			b=True
		self.assertEqual(True, b) 
		# remove temp directory 
		os.chdir(train_dir)
		shutil.rmtree(train_dir+'/'+directory)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_features(unittest.TestCase):
	'''
	tests featurization capabilities across all training scripts.

	- change settings.json to include every type of featurization
	- change back to default settings.
	'''
#### ##### ##### ##### ##### ##### ##### ##### ##### #####

	# audio features
	def test_audio_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir, default_audio_features=default_audio_features):
		directory='audio_features'
		folder=train_dir+'/'+directory
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		# put test audio file in directory 
		shutil.copy(cur_dir+'/test_audio.wav', folder+'/test_audio.wav')

		os.chdir(features_dir+'/audio_features/')
		features_list=default_audio_features

		# features that don't work
		# ['audioset_features', 'mixed_features', 'myprosody_features']

		for i in range(len(features_list)):
			print('------------------------------')
			print('FEATURIZING - %s'%(features_list[i].upper()))
			print('------------------------------')
			os.system('python3 featurize.py %s %s'%(folder, features_list[i]))

		# now that we have the folder let's check if the array has all the features
		os.chdir(folder)
		gopen=open('test_audio.json','r')
		g=json.load(gopen)
		features=g['features']['audio']
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

		notcount=str(notcount) + ' failed during featurization'

		self.assertEqual(True, b, notcount) 
		os.chdir(train_dir)
		shutil.rmtree(directory)
		
	# text features
	def test_text_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir, default_text_features=default_text_features):
		directory='text_features'
		folder=train_dir+'/'+directory
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		# put test audio file in directory 
		shutil.copy(cur_dir+'/test_text.txt', folder+'/test_text.txt')

		os.chdir(features_dir+'/text_features/')
		features_list=default_text_features

		# features that don't work
		# features = ['w2vec_features']

		for i in range(len(features_list)):
			print('------------------------------')
			print('FEATURIZING - %s'%(features_list[i].upper()))
			print('------------------------------')
			os.system('python3 featurize.py %s %s'%(folder, features_list[i]))

		# now that we have the folder let's check if the array has all the features
		os.chdir(folder)
		gopen=open('test_text.json')
		g=json.load(gopen)
		features=g['features']['text']
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

		notcount=str(notcount) + ' failed during featurization'

		self.assertEqual(True, b, notcount) 
		os.chdir(train_dir)
		shutil.rmtree(directory)

	# image features
	def test_image_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir, default_image_features=default_image_features):
		directory='image_features'
		folder=train_dir+'/'+directory
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		# put test audio file in directory 
		shutil.copy(cur_dir+'/test_image.png', folder+'/test_image.png')

		os.chdir(features_dir+'/image_features/')
		features_list=default_image_features

		# features that dont work 
		# features=['VGG19 features']

		for i in range(len(features_list)):
			print('------------------------------')
			print('FEATURIZING - %s'%(features_list[i].upper()))
			print('------------------------------')
			os.system('python3 featurize.py %s %s'%(folder, features_list[i]))

		# now that we have the folder let's check if the array has all the features
		os.chdir(folder)
		gopen=open('test_image.json','r')
		g=json.load(gopen)
		features=g['features']['image']
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

		notcount=str(notcount) + ' failed during featurization'

		self.assertEqual(True, b, notcount) 
		os.chdir(train_dir)
		shutil.rmtree(directory)

	# video features
	def test_video_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir, default_video_features=default_video_features):
		directory='video_features'
		folder=train_dir+'/'+directory
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		# put test audio file in directory 
		shutil.copy(cur_dir+'/test_video.mp4', folder+'/test_video.mp4')

		os.chdir(features_dir+'/video_features/')
		features_list=default_video_features

		# features that don't work 
		# features = ['y8m_features']

		for i in range(len(features_list)):
			print('------------------------------')
			print('FEATURIZING - %s'%(features_list[i].upper()))
			print('------------------------------')
			os.system('python3 featurize.py %s %s'%(folder, features_list[i]))

		# now that we have the folder let's check if the array has all the features
		os.chdir(folder)
		listdir=os.listdir()
		for i in range(len(listdir)):
                        if listdir[i].endswith('.mp4'):
                                videofile=listdir[i]
		gopen=open(listdir[i][0:-4]+'.json','r')
		g=json.load(gopen)
		features=g['features']['video']
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

		notcount=str(notcount) + ' failed during featurization'

		self.assertEqual(True, b, notcount) 
		os.chdir(train_dir)
		shutil.rmtree(directory)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_transcription(unittest.TestCase):
	'''
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
		os.chdir(train_dir)
		directory='audio_transcription'
		folder=train_dir+'/'+directory
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		# put test audio file in directory 
		shutil.copy(cur_dir+'/test_audio.wav', folder+'/test_audio.wav')

		os.chdir(features_dir+'/audio_features/')
		os.system('python3 featurize.py %s'%(folder))

		# now that we have the folder let's check if the array has all the features
		os.chdir(folder)
		g=json.load(open('test_audio.json'))
		transcripts=list(g['transcripts']['audio'])

		if default_audio_transcript[0] in transcripts:
			msg='success'
			b=True
		else:
			msg='failure'
			b=False 

		self.assertEqual(True, b, msg) 
		os.chdir(train_dir)
		shutil.rmtree(directory)

	# text transcription
	def test_text_transcription(self, default_text_transcript=default_text_transcript, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir):
		os.chdir(train_dir)
		directory='text_transcription'
		folder=train_dir+'/'+directory
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		# put test audio file in directory 
		shutil.copy(cur_dir+'/test_text.txt', folder+'/test_text.txt')
		os.chdir(features_dir+'/text_features/')
		os.system('python3 featurize.py %s'%(folder))

		# now that we have the folder let's check if the array has all the features
		os.chdir(folder)
		g=json.load(open('test_text.json'))
		transcripts=list(g['transcripts']['text'])

		if default_text_transcript[0] in transcripts:
			msg='success'
			b=True
		else:
			msg='fail'
			b=False 

		self.assertEqual(True, b, msg) 
		os.chdir(train_dir)
		shutil.rmtree(directory)

	# image transcription
	def test_image_transcription(self, default_image_transcript=default_image_transcript, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir):
		os.chdir(train_dir)
		directory='image_transcription'
		folder=train_dir+'/'+directory
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		# put test audio file in directory 
		shutil.copy(cur_dir+'/test_image.png', folder+'/test_image.png')
		os.chdir(features_dir+'/image_features/')
		os.system('python3 featurize.py %s'%(folder))

		# now that we have the folder let's check if the array has all the features
		os.chdir(folder)
		g=json.load(open('test_image.json'))
		transcripts=list(g['transcripts']['image'])

		if default_image_transcript[0] in transcripts:
			msg='success'
			b=True
		else:
			msg='failure'
			b=False 

		self.assertEqual(True, b, msg) 
		os.chdir(train_dir)
		shutil.rmtree(directory)

	# video transcription
	def test_video_transcription(self, default_video_transcript=default_video_transcript, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir):
		os.chdir(train_dir)
		directory='video_transcription'
		folder=train_dir+'/'+directory
		os.chdir(train_dir)
		try:
			os.mkdir(directory)
		except:
			shutil.rmtree(directory)
			os.mkdir(directory)

		# put test audio file in directory 
		shutil.copy(cur_dir+'/test_video.mp4', folder+'/test_video.mp4')

		os.chdir(features_dir+'/video_features/')
		os.system('python3 featurize.py %s'%(folder))

		# now that we have the folder let's check if the array has all the features
		os.chdir(folder)
		listdir=os.listdir()
		g=json.load(open('test_video.json'))
		transcripts=list(g['transcripts']['video'])

		if default_video_transcript[0] in transcripts:
			msg='success'
			b=True
		else:
			msg='failure'
			b=False 

		self.assertEqual(True, b, msg) 
		os.chdir(train_dir)
		shutil.rmtree(directory)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_training(unittest.TestCase):
	'''
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
	FEATURIZATION AND LOADING TESTS

	# note we have to do a loop here to end where the end is 
	# 'audio.json' | 'text.json' | 'image.json' | 'video.json' | 'csv.json'
	# this is because the files are renamed to not have conflicts.
	# for example, if 'audio.wav' --> 'audio.json' and 'audio.mp4' --> 'audio.json',
	# both would have a conflicting name and would overwrite each other. 
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
			if listdir[i].endswith('.wav') and listdir[i][0:-4]+'.json' in listdir: 
				b=True
				break

		# now remove all the temp files 
		os.chdir(loadmodel_dir+'/%s_models'%(filetype))
		for i in range(len(tempfiles)):
			shutil.rmtree(tempfiles[i])

		self.assertEqual(True, b)

	def test_loadtext(self, load_dir=load_dir, cur_dir=cur_dir, loadmodel_dir=loadmodel_dir):
		filetype='text'
		testfile='test_text.txt'
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
			if listdir[i].endswith('.txt') and listdir[i][0:-4]+'.json' in listdir: 
				b=True
				break

		# now remove all the temp files 
		os.chdir(loadmodel_dir+'/%s_models'%(filetype))
		for i in range(len(tempfiles)):
			shutil.rmtree(tempfiles[i])

		self.assertEqual(True, b)

	def test_loadimage(self, load_dir=load_dir, cur_dir=cur_dir, loadmodel_dir=loadmodel_dir):
		filetype='image'
		testfile='test_image.png'
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
			if listdir[i].endswith('.png') and listdir[i][0:-4]+'.json' in listdir:
				b=True
				break

		# now remove all the temp files 
		os.chdir(loadmodel_dir+'/%s_models'%(filetype))
		for i in range(len(tempfiles)):
			shutil.rmtree(tempfiles[i])

		self.assertEqual(True, b)

	def test_loadvideo(self, load_dir=load_dir, cur_dir=cur_dir, loadmodel_dir=loadmodel_dir):
		filetype='video'
		testfile='test_video.mp4'
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
		print(listdir)
		for i in range(len(listdir)):
			if listdir[i].endswith('.mp4') and listdir[i][0:-4]+'.json' in listdir: 
				b=True
				break

		# now remove all the temp files 
		os.chdir(loadmodel_dir+'/%s_models'%(filetype))
		for i in range(len(tempfiles)):
			shutil.rmtree(tempfiles[i])

		self.assertEqual(True, b)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_preprocessing(unittest.TestCase):

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

class test_visualization(unittest.TestCase):
	'''
	test the visualization module with text sample data in the test directory
	'''

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

