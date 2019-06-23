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

def text_featurize(feature_set, transcript, glovemodel, w2vmodel, fastmodel):

	if feature_set == 'nltk_features':
		features, labels = nf.nltk_featurize(transcript)
	elif feature_set == 'spacy_features':
		features, labels = sf.spacy_featurize(transcript)
	elif feature_set == 'glove_features':
		features, labels=gf.glove_featurize(transcript, glovemodel)
	elif feature_set == 'w2v_features':
		features, labels=w2v.w2v_featurize(transcript, w2vmodel)
	elif feature_set == 'fast_features':
		features, labels=ff.fast_featurize(transcript, fastmodel)

	return features, labels 

# type in folder before downloading and loading large files.
foldername=sys.argv[1]

# get class label from folder name 
labelname=foldername.split('/')
if labelname[-1]=='':
	labelname=labelname[-2]
else:
	labelname=labelname[-1]

##################################################
##				   Main script  		    	##
##################################################

basedir=os.getcwd()

# directory=sys.argv[1]
settingsdir=prev_dir(basedir)
settingsdir=prev_dir(settingsdir)
settings=json.load(open(settingsdir+'/settings.json'))
os.chdir(basedir)
try:
	feature_sets=[sys.argv[2]]
except:
	feature_sets=settings['default_text_features']
default_text_transcriber='raw_text'

# can specify many types of features...
for j in range(len(feature_sets)):
	feature_set=feature_sets[j]

	if feature_set in ['nltk_features', 'spacy_features']:
		# save memory by not loading any models that are not necessary.
		glovemodel=[]
		w2vmodel=[]
		fastmodel=[]

	else:
		##################################################
		##				Load ML models					##
		##################################################

		# load GloVE model
		if feature_set == 'glove_features':
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
			w2vmodel=[]
			fastmodel=[]
		# load Google W2V model
		elif feature_set == 'w2v_features':
			if 'GoogleNews-vectors-negative300.bin' not in os.listdir(os.getcwd()+'/helpers'):
				print('downloading Google W2V model...')
				wget.download("http://neurolex.co/uploads/GoogleNews-vectors-negative300.bin", "./helpers/GoogleNews-vectors-negative300.bin")

			w2vmodelname = 'GoogleNews-vectors-negative300.bin'
			print('-----------------')
			print('loading Google W2V model...')
			w2vmodel = KeyedVectors.load_word2vec_format(os.getcwd()+'/helpers/'+w2vmodelname, binary=True)
			print('loaded Google W2V model...')
			glovemodel=[]
			fastmodel=[]

		# load facebook FastText model
		elif feature_set == 'fast_features':
			if 'wiki-news-300d-1M' not in os.listdir(os.getcwd()+'/helpers'):
				print('downloading Facebook FastText model...')
				wget.download("https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip", "./helpers/wiki-news-300d-1M.vec.zip")
				zip_ref = zipfile.ZipFile(os.getcwd()+'/helpers/wiki-news-300d-1M.vec.zip', 'r')
				zip_ref.extractall(os.getcwd()+'/helpers/wiki-news-300d-1M')
				zip_ref.close()

			print('-----------------')
			print('loading Facebook FastText model...')
			# Loading fasttext model 
			fastmodel = KeyedVectors.load_word2vec_format(os.getcwd()+'/helpers/wiki-news-300d-1M/wiki-news-300d-1M.vec')
			print('loaded Facebook FastText model...')
			glovemodel=[]
			w2vmodel=[]

# change to folder directory 
os.chdir(foldername)
listdir=os.listdir() 
cur_dir=os.getcwd()

# featurize all files accoridng to librosa featurize
for i in range(len(listdir)):
	if listdir[i][-4:] in ['.txt']:
		#try:
		sampletype='text'
		os.chdir(cur_dir)
		transcript=open(listdir[i]).read()

		# I think it's okay to assume audio less than a minute here...
		if listdir[i][0:-4]+'.json' not in listdir:

			# make new .JSON if it is not there with base array schema.
			basearray=make_features(sampletype)

			# assume text_transcribe==True and add to transcript list 
			transcript_list=basearray['transcripts']
			transcript_list['text'][default_text_transcriber]=transcript 
			basearray['transcripts']=transcript_list

			for j in range(len(feature_sets)):
				feature_set=feature_sets[j]
				# featurize the text file 
				features, labels = text_featurize(feature_set, transcript, glovemodel, w2vmodel, fastmodel)
				print(features)

				try:
					data={'features':features.tolist(),
						  'labels': labels}
				except:
					data={'features':features,
						  'labels': labels}

				text_features=basearray['features']['text']
				text_features[feature_set]=data
				basearray['features']['text']=text_features

			basearray['labels']=[foldername]
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()

		elif listdir[i][0:-4]+'.json' in listdir:
			
			# load base array 
			basearray=json.load(open(listdir[i][0:-4]+'.json'))

			# get transcript and update if necessary 
			transcript_list=basearray['transcripts']

			if default_text_transcriber not in list(transcript_list['text']):
				transcript_list['text'][default_audio_transcriber]=transcript 
				basearray['transcripts']=transcript_list

			for j in range(len(feature_sets)):
				feature_set=feature_sets[j]
				# re-featurize only if necessary 
				if feature_set not in list(basearray['features']['text']):

					features, labels = text_featurize(feature_set, transcript, glovemodel, w2vmodel, fastmodel)
					print(features)

					try:
						data={'features':features.tolist(),
							  'labels': labels}
					except:
						data={'features':features,
							  'labels': labels}

					basearray['features']['text'][feature_set]=data

			# only add the label if necessary 
			label_list=basearray['labels']
			if labelname not in label_list:
				label_list.append(labelname)
			basearray['labels']=label_list

			# overwrite existing .JSON
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()

		#except:
			#print('error')





