import os, json, wget
# import video_features as vf 
import y8m_features as yf
from gensim.models import KeyedVectors
import os, wget, zipfile 
import shutil

##################################################
##				Helper functions.    			##
##################################################

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

def prev_dir(directory):
	g=directory.split('/')
	# print(g)
	lastdir=g[len(g)-1]
	i1=directory.find(lastdir)
	directory=directory[0:i1]
	return directory

##################################################
##				   Main script  		    	##
##################################################

# directory=sys.argv[1]
basedir=os.getcwd()
help_dir=basedir+'/helpers'
prevdir=prev_dir(basedir)

audioset_dir=prevdir+'audio_features'
os.chdir(audioset_dir)
import audioset_features as af 
os.chdir(basedir)

haar_dir=prevdir+'image_features/helpers/haarcascades/'  
foldername=input('what is the name of the folder?')
os.chdir(foldername)
cur_dir=os.getcwd()
listdir=os.listdir() 

# feature_set='video_features'
feature_set='y8m_features'

##################################################
##				Download inception 		    	##
##################################################

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
		videofile=listdir[i]
		sampletype='csv'
		# I think it's okay to assume audio less than a minute here...
		if listdir[i][0:-4]+'.json' not in listdir:
			# make new .JSON if it is not there with base array schema.
			basearray=make_features(sampletype)
			video_features=basearray['features']['video']
			# features, labels, transcript = vf.video_featurize(videofile, cur_dir, haar_dir)
			features, labels, transcript = yf.y8m_featurize(videofile, cur_dir, help_dir, fast_model)

			try:
				data={'features':features.tolist(),
					  'labels': labels}
			except:
				data={'features':features,
					  'labels': labels}

			video_features[feature_set]=data
			basearray['features']['video']=video_features
			basearray['labels']=[foldername]
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()
		elif listdir[i][0:-4]+'.json' in listdir:
			# overwrite existing .JSON if it is there.
			basearray=json.load(open(listdir[i][0:-4]+'.json'))
			
			# probably makes sense just to transcribe here and push through all - something like this 
			# os.system('ffmpeg -i %s %s'%(listdir[i], listdir[i][0:-4]+'.wav'))
			# transcript=transcribe(listdir[i][0:-4]+'.wav')
			# os.remove(listdir[i][0:-4]+'.wav')
			
			# features, labels, transcript = vf.video_featurize(videofile, cur_dir, haar_dir)
			features, labels, transcript = vf.video_featurize(videofile, cur_dir, haar_dir)
			print(features)

			try:
				data={'features':features.tolist(),
					  'labels': labels}
			except:
				data={'features':features,
					  'labels': labels}

			basearray['features']['video'][feature_set]=data
			basearray['labels']=[foldername]
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()

		#except:
			#print('error')