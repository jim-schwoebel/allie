'''
Import all the featurization scripts and allow the user to customize what embedding that
they would like to use for modeling purposes.

AudioSet is the only embedding that is a little bit wierd, as it is normalized to the length
of each audio file. There are many ways around this issue (such as normalizing to the length 
of each second), however, I included all the original embeddings here in case the time series
information is useful to you.
'''

import json, os, sys, time
import numpy as np 
import helpers.transcribe as ts
import speech_recognition as sr
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
	
# import to get image feature script 
directory=os.getcwd()
prevdir=prev_dir(directory)
sys.path.append(prevdir+'/image_features')
haar_dir=prevdir+'/image_features/helpers/haarcascades'
import image_features as imf
sys.path.append(prevdir+'/text_features')
import nltk_features as nf 
os.chdir(directory)

################################################
##	    		Helper functions      		  ##
################################################

def transcribe(file, default_audio_transcriber, settingsdir):
	# create all transcription methods here
	print('%s transcribing: %s'%(default_audio_transcriber, file))

	# use the audio file as the audio source
	r = sr.Recognizer()
	transcript_engine = default_audio_transcriber

	with sr.AudioFile(file) as source:
		audio = r.record(source)  # read the entire audio file

	if transcript_engine == 'pocketsphinx':

		# recognize speech using Sphinx
		try:
			transcript= r.recognize_sphinx(audio)
		except sr.UnknownValueError:
			transcript=''
		except sr.RequestError as e:
			transcript=''

	elif transcript_engine == 'deepspeech_nodict':

		curdir=os.getcwd()
		os.chdir(settingsdir+'/features/audio_features/helpers')
		listdir=os.listdir()
		deepspeech_dir=os.getcwd()

		# download models if not in helper directory
		if 'deepspeech-0.7.0-models.pbmm' not in listdir:
			os.system('wget https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.pbmm')

		# initialize filenames
		textfile=file[0:-4]+'.txt'
		newaudio=file[0:-4]+'_newaudio.wav'
		
		if deepspeech_dir.endswith('/'):
			deepspeech_dir=deepspeech_dir[0:-1]

		# go back to main directory
		os.chdir(curdir)

		# convert audio file to 16000 Hz mono audio 
		os.system('ffmpeg -i "%s" -acodec pcm_s16le -ac 1 -ar 16000 "%s" -y'%(file, newaudio))
		command='deepspeech --model %s/deepspeech-0.7.0-models.pbmm --audio "%s" >> "%s"'%(deepspeech_dir, newaudio, textfile)
		print(command)
		os.system(command)

		# get transcript
		transcript=open(textfile).read().replace('\n','')

		# remove temporary files
		os.remove(textfile)
		os.remove(newaudio)

	elif transcript_engine == 'deepspeech_dict':

		curdir=os.getcwd()
		os.chdir(settingsdir+'/features/audio_features/helpers')
		listdir=os.listdir()
		deepspeech_dir=os.getcwd()

		# download models if not in helper directory
		if 'deepspeech-0.7.0-models.pbmm' not in listdir:
			os.system('wget https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.pbmm')
		if 'deepspeech-0.7.0-models.scorer' not in listdir:
			os.system('wget https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.scorer')

		# initialize filenames
		textfile=file[0:-4]+'.txt'
		newaudio=file[0:-4]+'_newaudio.wav'
		
		if deepspeech_dir.endswith('/'):
			deepspeech_dir=deepspeech_dir[0:-1]

		# go back to main directory
		os.chdir(curdir)

		# convert audio file to 16000 Hz mono audio 
		os.system('ffmpeg -i "%s" -acodec pcm_s16le -ac 1 -ar 16000 "%s" -y'%(file, newaudio))
		command='deepspeech --model %s/deepspeech-0.7.0-models.pbmm --scorer %s/deepspeech-0.7.0-models.scorer --audio "%s" >> "%s"'%(deepspeech_dir, deepspeech_dir, newaudio, textfile)
		print(command)
		os.system(command)

		# get transcript
		transcript=open(textfile).read().replace('\n','')

		# remove temporary files
		os.remove(textfile)
		os.remove(newaudio)

	elif transcript_engine == 'google':

		# recognize speech using Google Speech Recognition
			# for testing purposes, we're just using the default API key
			# to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
			# instead of `r.recognize_google(audio)`

		# recognize speech using Google Cloud Speech
		GOOGLE_CLOUD_SPEECH_CREDENTIALS = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
		print(GOOGLE_CLOUD_SPEECH_CREDENTIALS)

		try:
			transcript=r.recognize_google_cloud(audio, credentials_json=open(GOOGLE_CLOUD_SPEECH_CREDENTIALS).read())
		except sr.UnknownValueError:
			transcript=''
		except sr.RequestError as e:
			transcript=''

	elif transcript_engine == 'wit':

		# recognize speech using Wit.ai
		WIT_AI_KEY = os.environ['WIT_AI_KEY']

		try:
			transcript=r.recognize_wit(audio, key=WIT_AI_KEY)
		except sr.UnknownValueError:
			transcript=''
		except sr.RequestError as e:
			transcript=''

	elif transcript_engine == 'azure':

		# recognize speech using Microsoft Azure Speech
		AZURE_SPEECH_KEY = os.environ['AZURE_SPEECH_KEY']
		print(AZURE_SPEECH_KEY)
		try:
			transcript=r.recognize_azure(audio, key=AZURE_SPEECH_KEY)
		except sr.UnknownValueError:
			transcript=''
		except sr.RequestError as e:
			transcript=''

	elif transcript_engine == 'bing':
		# recognize speech using Microsoft Bing Voice Recognition
		BING_KEY = os.environ['BING_KEY']
		try:
			transcript=r.recognize_bing(audio, key=BING_KEY)
		except sr.UnknownValueError:
			transcript=''
		except sr.RequestError as e:
			transcript=''

	elif transcript_engine == 'houndify':
		# recognize speech using Houndify
		HOUNDIFY_CLIENT_ID = os.environ['HOUNDIFY_CLIENT_ID']  
		HOUNDIFY_CLIENT_KEY = os.environ['HOUNDIFY_CLIENT_KEY']  
		try:
			transcript=r.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY)
		except sr.UnknownValueError:
			transcript=''
		except sr.RequestError as e:
			transcript=''

	elif transcript_engine == 'ibm':
		# recognize speech using IBM Speech to Text
		IBM_USERNAME = os.environ['IBM_USERNAME']
		IBM_PASSWORD = os.environ['IBM_PASSWORD']

		try:
			transcript=r.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD)
		except sr.UnknownValueError:
			transcript=''
		except sr.RequestError as e:
			transcript=''

	else:
		print('no transcription engine specified')
		transcript=''

	# show transcript
	print(transcript_engine.upper())
	print('--> '+ transcript)

	return transcript 

def audio_featurize(feature_set, audiofile, transcript):

	# long conditional on all the types of features that can happen and featurizes accordingly.
	if feature_set == 'audioset_features':
		features, labels = af.audioset_featurize(audiofile, basedir, foldername)
	elif feature_set == 'audiotext_features':
		features, labels = atf.audiotext_featurize(audiofile, transcript)
	elif feature_set == 'sox_features':
		features, labels = soxf.sox_featurize(audiofile)
	elif feature_set == 'sa_features':
		features, labels = saf.sa_featurize(audiofile)
	elif feature_set == 'pyaudio_features':
		features, labels = pf.pyaudio_featurize(audiofile, basedir)
	elif feature_set == 'librosa_features':
		features, labels = lf.librosa_featurize(audiofile, False)
	elif feature_set == 'meta_features':
		features, labels = mf.meta_featurize(audiofile, cur_dir, help_dir)
	elif feature_set == 'mixed_features':
		features, labels = mixf.mixed_featurize(audiofile, transcript, help_dir)
	elif feature_set == 'myprosody_features':
		print('Myprosody features are coming soon!! Currently debugging this feature set.')
		# features, labels = mpf.myprosody_featurize(audiofile, cur_dir, help_dir)
	elif feature_set == 'nltk_features':
		features, labels = nf.nltk_featurize(transcript)
	elif feature_set == 'opensmile_features':
		features, labels = osm.opensmile_featurize(audiofile, basedir, 'GeMAPSv01a.conf')
	elif feature_set == 'praat_features':
		features, labels = prf.praat_featurize(audiofile)
	elif feature_set == 'prosody_features':
		features, labels = prosf.prosody_featurize(audiofile, 20)
	elif feature_set == 'pspeech_features':
		features, labels = psf.pspeech_featurize(audiofile)
	elif feature_set == 'pspeechtime_features':
                features, labels = pstf.pspeech_featurize(audiofile)
	elif feature_set == 'pyaudiolex_features':
		features, labels = pyaudiolex.pyaudiolex_featurize(audiofile)
	elif feature_set == 'pyworld_features':
		features, labels = pywf.pyworld_featurize(audiofile)
	elif feature_set == 'specimage_features':
		features, labels = sif.specimage_featurize(audiofile,cur_dir, haar_dir)
	elif feature_set == 'specimage2_features':
		features, labels = sif2.specimage2_featurize(audiofile, cur_dir, haar_dir)
	elif feature_set == 'spectrogram_features':
		features, labels= specf.spectrogram_featurize(audiofile)
	elif feature_set == 'speechmetrics_features':
		features, labels=smf.speechmetrics_featurize(audiofile)
	elif feature_set == 'standard_features':
		features, labels = sf.standard_featurize(audiofile)

	# make sure all the features do not have any infinity or NaN
	features=np.nan_to_num(np.array(features))
	features=features.tolist()

	return features, labels 

def make_features(sampletype):

	# only add labels when we have actual labels.
	features={'audio':dict(),
			  'text': dict(),
			  'image':dict(),
			  'video':dict(),
			  'csv': dict(),
			  }

	transcripts={'audio': dict(),
				 'text': dict(),
				 'image': dict(),
				 'video': dict(),
				 'csv': dict()}
	models={'audio': dict(),
		 'text': dict(),
		 'image': dict(),
		 'video': dict(),
		 'csv': dict()}
	
	data={'sampletype': sampletype,
		  'transcripts': transcripts,
		  'features': features,
	      	  'models': models,
		  'labels': []}

	return data


################################################
##	    		Load main settings    		  ##
################################################

# directory=sys.argv[1]
basedir=os.getcwd()
settingsdir=prev_dir(basedir)
settingsdir=prev_dir(settingsdir)
settings=json.load(open(settingsdir+'/settings.json'))
os.chdir(basedir)

audio_transcribe=settings['transcribe_audio']
default_audio_transcribers=settings['default_audio_transcriber']
try:
	# assume 1 type of feature_set 
	feature_sets=[sys.argv[2]]
except:
	# if none provided in command line, then load deafult features 
	feature_sets=settings['default_audio_features']

# ^^ feature set set by settings.json above ^^ 
# here are some options below to give an example of what is possible
# feature_sets=['librosa_features']
# feature_sets=['standard_features']
# feature_sets=['audioset_features']
# feature_sets=['sox_features']
# feature_sets=['sa_features']
# feature_sets=['pyaudio_features']
# feature_sets=['spectrogram_features']
# feature_sets= ['meta_features']
# feature_sets=['praat_features']
# feature_sets=['pspeech_features']
# feature_sets=['specimage_features']
# feature_sets=['specimage2_features']
# feature_sets=['myprosody_features']
# feature_sets= ['nltk_features']
# feature_sets= ['mixed_features']
# feature_sets= ['audiotext_features']
# feature_sets=['librosa_features', 'standard_features', 'audioset_features', 'sox_features',
		# 	  'sa_features', 'pyaudio_features', 'spectrogram_features', 'meta_features',
		# 	  'praat_features', 'pspeech_features', 'specimage_features', 'specimage2_features',
		# 	  'myprosody_features', 'nltk_features', 'mixed_features', 'audiotext_features']
		
################################################
##	    	Import According to settings      ##
################################################

# only load the relevant featuresets for featurization to save memory
if 'audioset_features' in feature_sets:
	import audioset_features as af 
if 'audiotext_features' in feature_sets:
	import audiotext_features as atf
if 'sox_features' in feature_sets:
	import sox_features as soxf 
if 'pyaudio_features' in feature_sets:
	import pyaudio_features as pf 
if 'librosa_features' in feature_sets:
	import librosa_features as lf 
if 'meta_features' in feature_sets:
	import meta_features as mf 
	os.system('pip3 install scikit-learn==0.19.1')
if 'mixed_features' in feature_sets:
	import mixed_features as mixf
if 'myprosody_features' in feature_sets:
	pass
	# import myprosody_features as mpf
if 'opensmile_features' in feature_sets:
	import opensmile_features as osm
if 'pyaudiolex_features' in feature_sets:
	import pyaudiolex_features as pyaudiolex
if 'praat_features' in feature_sets:
	import praat_features as prf
if 'prosody_features' in feature_sets:
	import prosody_features as prosf
if 'pspeech_features' in feature_sets:
	import pspeech_features as psf
if 'pspeechtime_features' in feature_sets:
        import pspeechtime_features as pstf
if 'pyworld_features' in feature_sets:
	import pyworld_features as pywf
if 'sa_features' in feature_sets:
	import sa_features as saf
if 'specimage_features' in feature_sets:
	import specimage_features as sif
if 'specimage2_features' in feature_sets:
	import specimage2_features as sif2
if 'spectrogram_features' in feature_sets:
	import spectrogram_features as specf
if 'speechmetrics_features' in feature_sets:
	import speechmetrics_features as smf
if 'standard_features' in feature_sets:
	import standard_features as sf 

################################################
##	   		Get featurization folder     	  ##
################################################

foldername=sys.argv[1]
os.chdir(foldername)
listdir=os.listdir() 
cur_dir=os.getcwd()
help_dir=basedir+'/helpers/'

# get class label from folder name 
labelname=foldername.split('/')
if labelname[-1]=='':
	labelname=labelname[-2]
else:
	labelname=labelname[-1]

################################################
##	    	Now go featurize!                 ##
################################################

# featurize all files accoridng to librosa featurize
for i in tqdm(range(len(listdir)), desc=labelname):
	if listdir[i][-4:] in ['.wav', '.mp3', '.m4a']:
		filename=listdir[i]
		if listdir[i][-4:]=='.m4a':
			os.system('ffmpeg -i %s %s'%(listdir[i], listdir[i][0:-4]+'.wav'))
			filename=listdir[i][0:-4]+'.wav'
			os.remove(listdir[i])

		try:
			os.chdir(foldername)
			sampletype='audio'

			if listdir[i][0:-4]+'.json' not in listdir:

				# make new .JSON if it is not there with base array schema.
				basearray=make_features(sampletype)

				# get the first audio transcriber and loop through transcript list
				if audio_transcribe==True:
					for j in range(len(default_audio_transcribers)):
						default_audio_transcriber=default_audio_transcribers[j]
						transcript = transcribe(filename, default_audio_transcriber, settingsdir)
						transcript_list=basearray['transcripts']
						transcript_list['audio'][default_audio_transcriber]=transcript 
						basearray['transcripts']=transcript_list
				else:
					transcript=''

				# featurize the audio file 
				for j in range(len(feature_sets)):
					feature_set=feature_sets[j]
					features, labels = audio_featurize(feature_set, filename, transcript)
					try:
						data={'features':features.tolist(),
							  'labels': labels}
					except:
						data={'features':features,
							  'labels': labels}
					print(features)
					audio_features=basearray['features']['audio']
					audio_features[feature_set]=data
					basearray['features']['audio']=audio_features
				
				basearray['labels']=[labelname]

				# write to .JSON 
				jsonfile=open(listdir[i][0:-4]+'.json','w')
				json.dump(basearray, jsonfile)
				jsonfile.close()

			elif listdir[i][0:-4]+'.json' in listdir:
				# load the .JSON file if it is there 
				basearray=json.load(open(listdir[i][0:-4]+'.json'))
				transcript_list=basearray['transcripts']

				# only transcribe if you need to (checks within schema)
				if audio_transcribe==True:
					for j in range(len(default_audio_transcribers)):

						# get the first audio transcriber and loop through transcript list
						default_audio_transcriber=default_audio_transcribers[j]

						if audio_transcribe==True and default_audio_transcriber not in list(transcript_list['audio']):
							transcript = transcribe(filename, default_audio_transcriber, settingsdir)
							transcript_list['audio'][default_audio_transcriber]=transcript 
							basearray['transcripts']=transcript_list
						elif audio_transcribe==True and default_audio_transcriber in list(transcript_list['audio']):
							transcript = transcript_list['audio'][default_audio_transcriber]
						else:
							transcript=''
				else:
					transcript=''
					
				# only re-featurize if necessary (checks if relevant feature embedding exists)
				for j in range(len(feature_sets)):
					feature_set=feature_sets[j]
					if feature_set not in list(basearray['features']['audio']):
						features, labels = audio_featurize(feature_set, filename, transcript)
						print(features)
						try:
							data={'features':features.tolist(),
								  'labels': labels}
						except:
							data={'features':features,
								  'labels': labels}
						
						basearray['features']['audio'][feature_set]=data

				# only add the label if necessary 
				label_list=basearray['labels']
				if labelname not in label_list:
					label_list.append(labelname)
				basearray['labels']=label_list
				transcript_list=basearray['transcripts']

				# overwrite .JSON 
				jsonfile=open(listdir[i][0:-4]+'.json','w')
				json.dump(basearray, jsonfile)
				jsonfile.close()

		except:
			print('error')
	
# now reload the old scikit-learn
if 'meta_features' in feature_sets:
	import meta_features as mf 
	os.system('pip3 install scikit-learn==0.22.2.post1')
