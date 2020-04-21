'''
Import all the featurization scripts and allow the user to customize what embedding that
they would like to use for modeling purposes.

AudioSet is the only embedding that is a little bit wierd, as it is normalized to the length
of each audio file. There are many ways around this issue (such as normalizing to the length
of each second), however, I included all the original embeddings here in case the time series
information is useful to you.
'''

import json, os, sys
import numpy as np
import helpers.transcribe as ts
from tqdm import tqdm
import ray

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
##                      Helper functions                  ##
################################################

def transcribe(file, default_audio_transcriber):
		# get transcript
		try:
				if file[-4:]=='.wav':
						transcript=ts.transcribe_sphinx(file)
				elif file[-4] == '.mp3':
						os.system('ffmpeg -i %s %s'%(file, file[0:-4]+'.wav'))
						transcript=ts.transcribe_sphinx(file)
						os.remove(file[-4:]+'.wav')
				else:
						transcript=file
		except:
				transcript=''

		return transcript

def audio_featurize(feature_set, audiofile, transcript):

		# long conditional on all the types of features that can happen and featurizes accordingly.
		if feature_set == 'librosa_features':
				features, labels = lf.librosa_featurize(audiofile, False)
		elif feature_set == 'standard_features':
				features, labels = sf.standard_featurize(audiofile)
		elif feature_set == 'audioset_features':
				features, labels = af.audioset_featurize(audiofile, basedir, foldername)
		elif feature_set == 'sox_features':
				features, labels = soxf.sox_featurize(audiofile)
		elif feature_set == 'sa_features':
				features, labels = saf.sa_featurize(audiofile)
		elif feature_set == 'pyaudio_features':
				features, labels = pf.pyaudio_featurize(audiofile, basedir)
		elif feature_set == 'spectrogram_features':
				features, labels= specf.spectrogram_featurize(audiofile)
		elif feature_set == 'meta_features':
				features, labels = mf.meta_featurize(audiofile, cur_dir, help_dir)
		elif feature_set == 'opensmile_features':
				features, labels = osm.opensmile_featurize(audiofile, basedir, 'GeMAPSv01a.conf')
		elif feature_set == 'praat_features':
				features, labels = prf.praat_featurize(audiofile)
		elif feature_set == 'pspeech_features':
				features, labels = psf.pspeech_featurize(audiofile)
		elif feature_set == 'specimage_features':
				features, labels = sif.specimage_featurize(audiofile,cur_dir, haar_dir)
		elif feature_set == 'specimage2_features':
				features, labels = sif2.specimage2_featurize(audiofile, cur_dir, haar_dir)
		elif feature_set == 'myprosody_features':
				print('Myprosody features are coming soon!! Currently debugging this feature set.')
				# features, labels = mpf.myprosody_featurize(audiofile, cur_dir, help_dir)
		elif feature_set == 'nltk_features':
				features, labels = nf.nltk_featurize(transcript)
		elif feature_set == 'mixed_features':
				features, labels = mixf.mixed_featurize(audiofile, transcript, help_dir)
		elif feature_set == 'audiotext_features':
				features, labels = atf.audiotext_featurize(audiofile, transcript)
		elif feature_set == 'prosody_features':
				features, labels = prosf.prosody_featurize(audiofile, 20)
		elif feature_set == 'pyworld_features':
				features, labels = pywf.pyworld_featurize(audiofile)

		# make sure all the features do not have any infinity or NaN
		features=np.nan_to_num(np.array(features))
		features=features.tolist()

		return features, labels

@ray.remote
def audio_parallel_featurize(ifilename, foldername, default_audio_transcriber, audio_transcribe, feature_sets):

	if ifilename[-4:]=='.m4a':
				os.system('ffmpeg -i %s %s'%(listdir[i], listdir[i][0:-4]+'.wav'))
				filename=ifilename[0:-4]+'.wav'
				os.remove(ifilename)

				try:
						os.chdir(foldername)
						sampletype='audio'

						if ifilename[0:-4]+'.json' not in listdir:

								# make new .JSON if it is not there with base array schema.
								basearray=make_features(sampletype)

								# get the audio transcript
								if audio_transcribe==True:
										transcript = transcribe(filename, default_audio_transcriber)
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
								jsonfile=open(ifilename[0:-4]+'.json','w')
								json.dump(basearray, jsonfile)
								jsonfile.close()

						elif ifilename[0:-4]+'.json' in listdir:
								# load the .JSON file if it is there
								basearray=json.load(open(ifilename[0:-4]+'.json'))
								transcript_list=basearray['transcripts']

								# only transcribe if you need to (checks within schema)
								if audio_transcribe==True and default_audio_transcriber not in list(transcript_list['audio']):
										transcript = transcribe(filename, default_audio_transcriber)
										transcript_list['audio'][default_audio_transcriber]=transcript
										basearray['transcripts']=transcript_list
								elif audio_transcribe==True and default_audio_transcriber in list(transcript_list['audio']):
										transcript = transcript_list['audio'][default_audio_transcriber]
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
								jsonfile=open(ifilename[0:-4]+'.json','w')
								json.dump(basearray, jsonfile)
								jsonfile.close()

				except:
						print('error')


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
##                      Load main settings                ##
################################################

# directory=sys.argv[1]
basedir=os.getcwd()
settingsdir=prev_dir(basedir)
settingsdir=prev_dir(settingsdir)
settings=json.load(open(settingsdir+'/settings.json'))
os.chdir(basedir)

audio_transcribe=settings['transcribe_audio']
default_audio_transcriber=settings['default_audio_transcriber']
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
				#         'sa_features', 'pyaudio_features', 'spectrogram_features', 'meta_features',
				#         'praat_features', 'pspeech_features', 'specimage_features', 'specimage2_features',
				#         'myprosody_features', 'nltk_features', 'mixed_features', 'audiotext_features']

################################################
##              Import According to settings      ##
################################################

# only load the relevant featuresets for featurization to save memory
if 'librosa_features' in feature_sets:
		import librosa_features as lf
if 'standard_features' in feature_sets:
		import standard_features as sf
if 'audioset_features' in feature_sets:
		import audioset_features as af
if 'sox_features' in feature_sets:
		import sox_features as soxf
if 'pyaudio_features' in feature_sets:
		import pyaudio_features as pf
if 'sa_features' in feature_sets:
		import sa_features as saf
if 'spectrogram_features' in feature_sets:
		import spectrogram_features as specf
if 'meta_features' in feature_sets:
		import meta_features as mf
if 'opensmile_features' in feature_sets:
		import opensmile_features as osm
if 'praat_features' in feature_sets:
		import praat_features as prf
if 'prosody_features' in feature_sets:
		import prosody_features as prosf
if 'pspeech_features' in feature_sets:
		import pspeech_features as psf
if 'specimage_features' in feature_sets:
		import specimage_features as sif
if 'specimage2_features' in feature_sets:
		import specimage2_features as sif2
if 'myprosody_features' in feature_sets:
		pass
		# import myprosody_features as mpf
if 'mixed_features' in feature_sets:
		import mixed_features as mixf
if 'audiotext_features' in feature_sets:
		import audiotext_features as atf
if 'pyworld_features' in feature_sets:
		import pyworld_features as pywf

################################################
##                      Get featurization folder          ##
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
##              Now go featurize!                 ##
################################################
ray.init()
# featurize all files accoridng to librosa featurize
for i in tqdm(range(len(listdir)), desc=labelname):
		if listdir[i][-4:] in ['.wav', '.mp3', '.m4a']:
				filename=listdir[i]
				audio_parallel_featurize.remote(ifilename, foldername, default_audio_transcriber, audio_transcribe, feature_sets)
