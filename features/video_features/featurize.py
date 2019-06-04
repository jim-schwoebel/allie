import os, json
import video_features as vf 

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
prevdir=prev_dir(basedir)
haar_dir=prevdir+'image_features/helpers/haarcascades/'

foldername=input('what is the name of the folder?')
os.chdir(foldername)
cur_dir=os.getcwd()
listdir=os.listdir() 

feature_set='video_features'
# feature_set='VGG16_features'
# feature_set='Inception_features'
# feature_set='Xception_features'
# feature_set='Resnet50_features'
# feature_set='VGG19_features'

# featurize all files accoridng to librosa featurize
for i in range(len(listdir)):

	# make audio file into spectrogram and analyze those images if audio file
	if listdir[i][-4:] in ['.mp4']:
		#try:
		videofile=listdir[i]
			
		# I think it's okay to assume audio less than a minute here...
		if listdir[i][0:-4]+'.json' not in listdir:
			# make new .JSON if it is not there with base array schema.
			basearray=make_features()
			video_features=basearray['features']['video']
			features, labels, transcript = vf.video_featurize(videofile, cur_dir, haar_dir)

			print(features)

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