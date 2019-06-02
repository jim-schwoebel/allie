'''
Import all the featurization scripts and allow the user to customize what embedding that
they would like to use for modeling purposes.

AudioSet is the only embedding that is a little bit wierd, as it is normalized to the length
of each audio file. There are many ways around this issue (such as normalizing to the length 
of each second), however, I included all the original embeddings here in case the time series
information is useful to you.
'''
##################################################
##				Import statements   			##
##################################################

# scripts 
import nltk_features as nf
import spacy_features as sf
import glove_features as gf 
import fast_features as ff
import helpers.transcribe as ts

# other import stuff 
import json, os, sys
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
import os, wget, zipfile 
import shutil

##################################################
##				Helper functions.    			##
##################################################

def transcribe(file):
	# get transcript 
	if file[-4:]=='.wav':
		transcript=ts.transcribe_sphinx(file)
	elif file[-4] == '.mp3':
		os.system('ffmpeg -i %s %s'%(file, file[0:-4]+'.wav'))
		transcript=ts.transcribe_sphinx(file)
		os.remove(file[-4:]+'.wav')
	else:
		transcript=file

	transcript={'transcript': transcript,
				'transcript_type': 'pocketsphinx'}

	return transcript 

def make_features():

	# only add labels when we have actual labels.
	features={'audio':dict(),
			  'text': dict(),
			  'image':dict(),
			  'video':dict(),
			  }

	data={'features': features,
		  'labels': []}

	return data


##################################################
##				Load ML models					##
##################################################

# load GloVE model

if 'glove.6B' not in os.listdir(os.getcwd()+'/helpers'):
	curdir=os.getcwd()
	print('downloading GloVe model...')
	wget.download("http://neurolex.co/uploads/glove.6B.zip", "./helpers/glove.6B.zip")
	print('extracting GloVe model')
	zip_ref = zipfile.ZipFile(os.getcwd()+'/helpers/glove.6B.zip', 'r')
	zip_ref.extractall(os.getcwd()+'/helpers/glove.6B')
	zip_ref.close()
	os.chdir(os.getcwd()+'/helpers/glove.6B')
	glove_input_file = 'glove.6B.100d.txt'
	word2vec_output_file = 'glove.6B.100d.txt.word2vec'
	glove2word2vec(glove_input_file, word2vec_output_file)
	os.chdir(curdir)

glovemodelname = 'glove.6B.100d.txt.word2vec'
print('-----------------')
print('loading GloVe model...')
glovemodel = KeyedVectors.load_word2vec_format(os.getcwd()+'/helpers/glove.6B/'+glovemodelname, binary=False)
print('loaded GloVe model...')

# load Google W2V model

if 'GoogleNews-vectors-negative300.bin' not in os.listdir(os.getcwd()+'/helpers'):
	print('downloading Google W2V model...')
	wget.download("http://neurolex.co/uploads/GoogleNews-vectors-negative300.bin", "./helpers/GoogleNews-vectors-negative300.bin")

w2vmodelname = 'GoogleNews-vectors-negative300.bin'
print('-----------------')
print('loading Google W2V model...')
w2vmodel = KeyedVectors.load_word2vec_format(os.getcwd()+'/helpers/'+w2vmodelname, binary=True)
print('loaded Google W2V model...')

# load facebook FastText model

if 'wiki-news-300d-1M' not in os.listdir(os.getcwd()+'/helpers'):
	print('downloading Facebook FastText model...')
	wget.download("https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip")
	zip_ref = zipfile.ZipFile(os.getcwd()+'/helpers/wiki-news-300d-1M.vec.zip', 'r')
	zip_ref.extractall(os.getcwd()+'/helpers/wiki-news-300d-1M')
	zip_ref.close()

print('-----------------')
print('loading Facebook FastText model...')
# Loading fasttext model 
fastmodel = KeyedVectors.load_word2vec_format('/helpers/wiki-news-300d-1M/wiki-news-300d-1M.vec')
print('loaded Facebook FastText model...')

##################################################
##				   Main script  		    	##
##################################################

# directory=sys.argv[1]
basedir=os.getcwd()
foldername=input('what is the name of the folder?')
os.chdir(foldername)
listdir=os.listdir() 

# feature_set='nltk_features'
# feature_set='spacy_features'
# feature_set='glove_features'
# feature_set='w2vec_features'
feature_set='fast_features'

# featurize all files accoridng to librosa featurize
for i in range(len(listdir)):
	if listdir[i][-4:] in ['.wav', '.mp3']:
		#try:

		# I think it's okay to assume audio less than a minute here...
		if listdir[i][0:-4]+'.json' not in listdir:
			# make new .JSON if it is not there with base array schema.
			basearray=make_features()
			text_features=basearray['features']['text']
			transcript=transcribe(listdir[i])
			basearray['transcript']=transcript
			transcript=transcript['transcript']
			
			# features, labels = nf.nltk_featurize(transcript)
			# features, labels = sf.spacy_featurize(transcript)
			# features, labels=gf.glove_featurize(transcript, glovemodel)
			# features, labels=w2v.w2v_featurize(transcript, w2vmodel)
			features, labels=ff.fast_featurize(transcript, fastmodel)

			print(features)

			try:
				data={'features':features.tolist(),
					  'labels': labels}
			except:
				data={'features':features,
					  'labels': labels}

			text_features[feature_set]=data
			basearray['features']['text']=text_features
			basearray['labels']=[foldername]
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()
		elif listdir[i][0:-4]+'.json' in listdir:
			# overwrite existing .JSON if it is there.
			basearray=json.load(open(listdir[i][0:-4]+'.json'))

			try: 
				transcript=basearray['transcript']['transcript']
			except:
				transcript=transcribe(listdir[i])
				basearray['transcript']=transcript
				transcript=transcript['transcript']


			# features, labels = nf.nltk_featurize(transcript)
			# features, labels = sf.spacy_featurize(transcript)
			# features, labels=gf.glove_featurize(transcript, glovemodel)
			# features, labels=w2v.w2v_featurize(transcript, w2vmodel)
			features, labels=ff.fast_featurize(transcript, fastmodel)

			print(features)

			try:
				data={'features':features.tolist(),
					  'labels': labels}
			except:
				data={'features':features,
					  'labels': labels}

			basearray['features']['text'][feature_set]=data
			basearray['labels']=[foldername]
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()

		#except:
			#print('error')





