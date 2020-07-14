import os, json, wget, uuid
import numpy as np
from gensim.models import KeyedVectors
import os, wget, zipfile, sys
import shutil
from tqdm import tqdm

##################################################
##				Helper functions.    			##
##################################################

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

def video_featurize(feature_set, videofile, cur_dir, haar_dir, help_dir, fast_model):

	# long conditional on all the types of features that can happen and featurizes accordingly.
	if feature_set == 'video_features':
		features, labels, audio_transcript, image_transcript = vf.video_featurize(videofile, cur_dir, haar_dir)
	elif feature_set == 'y8m_features':
		features, labels, audio_transcript, image_transcript = yf.y8m_featurize(videofile, cur_dir, help_dir, fast_model)

	# make sure all the features do not have any infinity or NaN
	features=np.nan_to_num(np.array(features))
	features=features.tolist()

	return features, labels, audio_transcript, image_transcript 

def video_transcribe(default_video_transcriber, videofile):
	# this is a placeholder function now
	return ''

def audio_transcribe(default_audio_transcriber, audiofile):
	# this is a placeholder function now until we have more audio transcription engines
	return ''

##################################################
##				   Main script  		    	##
##################################################

# directory=sys.argv[1]
basedir=os.getcwd()
help_dir=basedir+'/helpers'
prevdir=prev_dir(basedir)
sys.path.append(prevdir)
from standard_array import make_features

# audioset_dir=prevdir+'/audio_features'
# os.chdir(audioset_dir)
# import audioset_features as af 
# os.chdir(basedir)

haar_dir=prevdir+'/image_features/helpers/haarcascades/'  

# get settings 
settingsdir=prev_dir(basedir)
settingsdir=prev_dir(settingsdir)
settings=json.load(open(settingsdir+'/settings.json'))
os.chdir(basedir)

# load settings
audio_transcribe_setting=settings['transcribe_audio']
video_transcribe_setting=settings['transcribe_video']
default_audio_transcriber=settings['default_audio_transcriber']
default_video_transcriber=settings['default_video_transcriber']
try:
	feature_sets=[sys.argv[2]]
except:
	feature_sets=settings['default_video_features']

# import proper feature sets from database 
if 'video_features' in feature_sets:
	import video_features as vf 
if 'y8m_features' in feature_sets:
	import y8m_features as yf

# change to video folder
foldername=sys.argv[1]
os.chdir(foldername)
cur_dir=os.getcwd()
listdir=os.listdir() 
os.chdir(cur_dir)

# get class label from folder name 
labelname=foldername.split('/')
if labelname[-1]=='':
	labelname=labelname[-2]
else:
	labelname=labelname[-1]
	
##################################################
##				Download inception 		    	##
##################################################

for j in range(len(feature_sets)):
	feature_set=feature_sets[j]
	if feature_set == 'video_features':
		fast_model=[]

	elif feature_set == 'y8m_features':

		if 'inception-2015-12-05.tgz' not in os.listdir(basedir+'/helpers'):
		    os.chdir(basedir+'/helpers')
		    filename = wget.download('http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz')
		    filename=wget.download('http://data.yt8m.org/yt8m_pca.tgz')
		    os.system('tar zxvf inception-2015-12-05.tgz')
		    os.system('tar zxvf yt8m_pca.tgz')
		    os.chdir(cur_dir)

		# load in FAST model 
		os.chdir(prevdir+'/text_features')
		if 'wiki-news-300d-1M' not in os.listdir(os.getcwd()+'/helpers'):
			print('downloading Facebook FastText model...')
			wget.download("https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip", "./helpers/wiki-news-300d-1M.vec.zip")
			zip_ref = zipfile.ZipFile(os.getcwd()+'/helpers/wiki-news-300d-1M.vec.zip', 'r')
			zip_ref.extractall(os.getcwd()+'/helpers/wiki-news-300d-1M')
			zip_ref.close()

		print('-----------------')
		print('loading Facebook FastText model...')
		# Loading fasttext model 
		fast_model = KeyedVectors.load_word2vec_format(os.getcwd()+'/helpers/wiki-news-300d-1M/wiki-news-300d-1M.vec')
		print('loaded Facebook FastText model...')
		os.chdir(cur_dir)

###################################################

# featurize all files accoridng to librosa featurize
for i in tqdm(range(len(listdir)), desc=labelname):

	# make audio file into spectrogram and analyze those images if audio file
	if listdir[i][-4:] in ['.mp4']:
		try:
			sampletype='video'
			videofile=listdir[i]
			
			# I think it's okay to assume audio less than a minute here...
			if listdir[i][0:-4]+'.json' not in listdir:

				# rename to avoid conflicts
				# newfile=str(uuid.uuid4())+'.mp4'
				# os.rename(listdir[i], newfile)
				# videofile=newfile
				
				# make new .JSON if it is not there with base array schema.
				basearray=make_features(sampletype)

				# get features and add label 
				for j in range(len(feature_sets)):
					feature_set=feature_sets[j]
					video_features=basearray['features']['video']
					features, labels, audio_transcript, video_transcript = video_featurize(feature_set, videofile, cur_dir, haar_dir, help_dir, fast_model)
					
					try:
						data={'features':features.tolist(),
							  'labels': labels}
					except:
						data={'features':features,
							  'labels': labels}

					video_features[feature_set]=data
					basearray['features']['video']=video_features

				basearray['labels']=[labelname]

				# only add transcripts in schema if they are true 
				transcript_list=basearray['transcripts']

				# video transcription setting
				if video_transcribe_setting == True:	
					for j in range(len(default_video_transcriber)):
						video_transcriber=default_video_transcriber[j]
						if video_transcriber=='tesseract (averaged over frames)':
							transcript_list['video'][video_transcriber] = video_transcript
						else:
							print('cannot transcribe video file, as the %s transcriber is not supported'%(video_transcriber.upper()))
				
				# audio transcriber setting
				if audio_transcribe_setting == True:
					for j in range(len(default_audio_transcriber)):
						audio_transcriber=default_audio_transcriber[j]
						if audio_transcriber == 'pocketsphinx':
							transcript_list['audio'][audio_transcriber] = audio_transcript 
						else:
							print('cannot transcribe audio file, as the %s transcriber is not supported'%(audio_transcriber.upper()))
				
				# update transcript list
				basearray['transcripts']=transcript_list

				# write to .JSON 
				jsonfile=open(videofile[0:-4]+'.json','w')
				json.dump(basearray, jsonfile)
				jsonfile.close()

			elif listdir[i][0:-4]+'.json' in listdir:
				# load the .JSON file if it is there 
				basearray=json.load(open(listdir[i][0:-4]+'.json'))
				video_features=basearray['features']['video']

				# featurizes/labels only if necessary (skips if feature embedding there)
				for j in range(len(feature_sets)):
					feature_set=feature_sets[j]
					if feature_set not in list(video_features):
						features, labels, audio_transcript, video_transcript = video_featurize(feature_set, videofile, cur_dir, haar_dir, help_dir, fast_model)
						print(features)
						try:
							data={'features':features.tolist(),
								  'labels': labels}
						except:
							data={'features':features,
								  'labels': labels}

						video_features[feature_set]=data
						basearray['features']['video']=video_features

				# make transcript additions, as necessary 
				transcript_list=basearray['transcripts']

				if video_transcribe_setting == True:
					for j in range(len(default_video_transcriber)):
						default_video_transcriber=default_video_transcriber[j]
						if video_transcriber not in list(transcript_list):
							if video_transcriber == 'tesseract (averaged over frames)':
								transcript_list['video'][video_transcriber] = video_transcript
							else:
								print('cannot transcribe video file, as the %s transcriber is not supported'%(video_transcriber.upper()))
				
				if audio_transcribe_setting == True:
					for j in range(len(default_audio_transcriber)):
						audio_transcriber=default_audio_transcriber[j]
						if audio_transcriber not in list(transcript_list):	
							if audio_transcriber == 'pocketsphinx':
								transcript_list['audio'][audio_transcriber] = audio_transcript
							else:
								print('cannot transcribe audio file, as the %s transcriber is not supported'%(audio_transcriber.upper()))
				
				basearray['transcripts']=transcript_list

				# add appropriate labels only if they are new labels 
				label_list=basearray['labels']
				if labelname not in label_list:
					label_list.append(labelname)
				basearray['labels']=label_list

				# overwrite .JSON 
				jsonfile=open(videofile[0:-4]+'.json','w')
				json.dump(basearray, jsonfile)
				jsonfile.close()

		except:
			print('error - already featurized %s'%(videofile.upper()))
