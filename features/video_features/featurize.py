import os, json, wget
import video_features as vf 
# import y8m_features as yf
from gensim.models import KeyedVectors
import os, wget, zipfile 
import shutil

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

def audio_transcribe(transcript, default_audio_transcriber):

	# 3 types of transcripts = audio, text, and image 
	transcript_dict={'transcript': transcript,
					'transcript_type': 'audio',
					'audio_transcriber': default_audio_transcriber}

	return transcript_dict, transcript 

def video_transcribe2(transcript, default_video_transcriber):

	# 3 types of transcripts = audio, text, and image 
	transcript_dict={'transcript': str(transcript),
					'transcript_type': 'video',
					'video_transcriber': default_video_transcriber}

	return transcript_dict, transcript 

def make_features(sampletype):

	# only add labels when we have actual labels.
	features={'audio':dict(),
			  'text': dict(),
			  'image':dict(),
			  'video':dict(),
			  'csv': dict(),
			  }

	data={'sampletype': sampletype,
		  'transcripts': [],
		  'features': features,
		  'labels': []}

	return data

def video_featurize(feature_set, videofile, cur_dir, haar_dir, help_dir, fast_model):

	# long conditional on all the types of features that can happen and featurizes accordingly.
	if feature_set == 'video_features':
		features, labels, audio_transcript, image_transcript = vf.video_featurize(videofile, cur_dir, haar_dir)
	elif feature_set == 'y8m_features':
		features, labels, audio_transcript, image_transcript = yf.y8m_featurize(videofile, cur_dir, help_dir, fast_model)

	return features, labels, audio_transcript, image_transcript 

##################################################
##				   Main script  		    	##
##################################################

# directory=sys.argv[1]
basedir=os.getcwd()
help_dir=basedir+'/helpers'
prevdir=prev_dir(basedir)

# audioset_dir=prevdir+'/audio_features'
# os.chdir(audioset_dir)
# import audioset_features as af 
# os.chdir(basedir)

haar_dir=prevdir+'/image_features/helpers/haarcascades/'  
foldername=sys.argv[1]
os.chdir(foldername)
cur_dir=os.getcwd()
listdir=os.listdir() 

# get settings 
settingsdir=prev_dir(basedir)
settingsdir=prev_dir(settingsdir)
settings=json.load(open(settingsdir+'/settings.json'))
os.chdir(basedir)

video_transcribe=settings['transcribe_videos']
default_audio_transcriber=settings['default_audio_transcriber']
default_video_transcriber=settings['default_video_transcriber']
feature_set=settings['default_video_features']
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
	os.chdir(prevdir+'text_features')
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
for i in range(len(listdir)):

	# make audio file into spectrogram and analyze those images if audio file
	if listdir[i][-4:] in ['.mp4']:
		#try:
		sampletype='video'
		videofile=listdir[i]
		features, labels, audio_transcript, video_transcript = video_featurize(feature_set, videofile, cur_dir, haar_dir, help_dir, fast_model)
		audio_transcript_dict, audio_transcript = audio_transcribe(audio_transcript, default_audio_transcriber)
		video_transcript_dict, video_transcript = video_transcribe2(video_transcript, default_video_transcriber)

		# I think it's okay to assume audio less than a minute here...
		if listdir[i][0:-4]+'.json' not in listdir:
			# make new .JSON if it is not there with base array schema.
			basearray=make_features(sampletype)
			video_features=basearray['features']['video']
			transcripts=[audio_transcript_dict, video_transcript_dict]

			try:
				data={'features':features.tolist(),
					  'labels': labels}
			except:
				data={'features':features,
					  'labels': labels}

			video_features[feature_set]=data
			basearray['features']['video']=video_features
			basearray['labels']=[foldername]
			basearray['transcripts']=transcripts
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()
		elif listdir[i][0:-4]+'.json' in listdir:
			# overwrite existing .JSON if it is there.
			basearray=json.load(open(listdir[i][0:-4]+'.json'))
			print(features)

			try:
				data={'features':features.tolist(),
					  'labels': labels}
			except:
				data={'features':features,
					  'labels': labels}

			basearray['features']['video'][feature_set]=data
			label_list=basearray['labels']
			if labelname not in label_list:
				label_list.append(labelname)
			basearray['labels']=label_list
			transcript_list=basearray['transcripts']
			transcript_list.append(audio_transcript_dict)
			transcript_list.append(video_transcript_dict)
			basearray['transcripts']=transcript_list
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()

		#except:
			#print('error')