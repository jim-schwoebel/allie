import VGG16_features as vf
import image_features as imf 
import helpers.audio_plot as ap 
import os, json

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

# # load GloVE model

# if 'glove.6B' not in os.listdir(os.getcwd()+'/helpers'):
# 	curdir=os.getcwd()
# 	print('downloading GloVe model...')
# 	wget.download("http://neurolex.co/uploads/glove.6B.zip", "./helpers/glove.6B.zip")
# 	print('extracting GloVe model')
# 	zip_ref = zipfile.ZipFile(os.getcwd()+'/helpers/glove.6B.zip', 'r')
# 	zip_ref.extractall(os.getcwd()+'/helpers/glove.6B')
# 	zip_ref.close()
# 	os.chdir(os.getcwd()+'/helpers/glove.6B')
# 	glove_input_file = 'glove.6B.100d.txt'
# 	word2vec_output_file = 'glove.6B.100d.txt.word2vec'
# 	glove2word2vec(glove_input_file, word2vec_output_file)
# 	os.chdir(curdir)

# glovemodelname = 'glove.6B.100d.txt.word2vec'
# print('-----------------')
# print('loading GloVe model...')
# glovemodel = KeyedVectors.load_word2vec_format(os.getcwd()+'/helpers/glove.6B/'+glovemodelname, binary=False)
# print('loaded GloVe model...')

# # load Google W2V model

# if 'GoogleNews-vectors-negative300.bin' not in os.listdir(os.getcwd()+'/helpers'):
# 	print('downloading Google W2V model...')
# 	wget.download("http://neurolex.co/uploads/GoogleNews-vectors-negative300.bin", "./helpers/GoogleNews-vectors-negative300.bin")

# w2vmodelname = 'GoogleNews-vectors-negative300.bin'
# print('-----------------')
# print('loading Google W2V model...')
# w2vmodel = KeyedVectors.load_word2vec_format(os.getcwd()+'/helpers/'+w2vmodelname, binary=True)
# print('loaded Google W2V model...')

# load facebook FastText model

##################################################
##				   Main script  		    	##
##################################################

# directory=sys.argv[1]
basedir=os.getcwd()
haar_dir=basedir+'/helpers/haarcascades'
foldername=input('what is the name of the folder?')
os.chdir(foldername)
cur_dir=os.getcwd()
listdir=os.listdir() 

feature_set='image_features'

# featurize all files accoridng to librosa featurize
for i in range(len(listdir)):
	if listdir[i][-4:] in ['.wav', '.mp3']:
		#try:

		# make audio file into spectrogram 
		if listdir[i][0:-4]+'.png' not in listdir:
			imgfile=ap.plot_spectrogram(listdir[i])
		else:
			imgfile=listdir[i][0:-4]+'.png'
			
		# I think it's okay to assume audio less than a minute here...
		if listdir[i][0:-4]+'.json' not in listdir:
			# make new .JSON if it is not there with base array schema.
			basearray=make_features()
			image_features=basearray['features']['image']
			features, labels=imf.image_featurize(cur_dir, haar_dir, imgfile)

			print(features)

			try:
				data={'features':features.tolist(),
					  'labels': labels}
			except:
				data={'features':features,
					  'labels': labels}

			image_features[feature_set]=data
			basearray['features']['image']=image_features
			basearray['labels']=[foldername]
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()
		elif listdir[i][0:-4]+'.json' in listdir:
			# overwrite existing .JSON if it is there.
			basearray=json.load(open(listdir[i][0:-4]+'.json'))
			features, labels=imf.image_featurize(cur_dir, haar_dir, imgfile)
			print(features)

			try:
				data={'features':features.tolist(),
					  'labels': labels}
			except:
				data={'features':features,
					  'labels': labels}

			basearray['features']['image'][feature_set]=data
			basearray['labels']=[foldername]
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()

		#except:
			#print('error')
