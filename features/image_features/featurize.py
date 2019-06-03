import VGG16_features as v16f
import image_features as imf 
import Inception_features as incf 
import Xception_features as xf 
import Resnet_features as rf 
import VGG19_features as v19f
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
##				   Main script  		    	##
##################################################

# directory=sys.argv[1]
basedir=os.getcwd()
haar_dir=basedir+'/helpers/haarcascades'
foldername=input('what is the name of the folder?')
os.chdir(foldername)
cur_dir=os.getcwd()
listdir=os.listdir() 

# feature_set='image_features'
# feature_set='VGG16_features'
# feature_set='Inception_features'
# feature_set='Xception_features'
# feature_set='Resnet50_features'
feature_set='VGG19_features'

# featurize all files accoridng to librosa featurize
for i in range(len(listdir)):

	# make audio file into spectrogram and analyze those images if audio file
	if listdir[i][-4:] in ['.wav', '.mp3']:
		try:
			if listdir[i][0:-4]+'.png' not in listdir:
				imgfile=ap.plot_spectrogram(listdir[i])
			else:
				imgfile=listdir[i][0:-4]+'.png'
				
			# I think it's okay to assume audio less than a minute here...
			if listdir[i][0:-4]+'.json' not in listdir:
				# make new .JSON if it is not there with base array schema.
				basearray=make_features()
				image_features=basearray['features']['image']

				# features, labels=imf.image_featurize(cur_dir, haar_dir, imgfile)
				# features, labels=v16f.VGG16_featurize(imgfile)
				# features, labels=incf.inception_featurize(imgfile)
				# features, labels=xf.xception_featurize(imgfile)
				# features, labels=rf.resnet_featurize(imgfile)
				features, labels=v16f.VGG16_featurize(imgfile)

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

				# features, labels=imf.image_featurize(cur_dir, haar_dir, imgfile)
				# features, labels=v16f.VGG16_featurize(imgfile)
				# features, labels=incf.inception_featurize(imgfile)
				# features, labels=xf.xception_featurize(imgfile)
				# features, labels=rf.resnet_featurize(imgfile)
				features, labels=v16f.VGG16_featurize(imgfile)

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

		except:
			print('error')

	elif listdir[i][-4:] in ['.jpg', '.png']:
		#try:
		imgfile=listdir[i]
			
		# I think it's okay to assume audio less than a minute here...
		if listdir[i][0:-4]+'.json' not in listdir:
			# make new .JSON if it is not there with base array schema.
			basearray=make_features()
			image_features=basearray['features']['image']

			# features, labels=imf.image_featurize(cur_dir, haar_dir, imgfile)
			# features, labels=v16f.VGG16_featurize(imgfile)
			# features, labels=incf.inception_featurize(imgfile)
			# features, labels=xf.xception_featurize(imgfile)
			# features, labels=rf.resnet_featurize(imgfile)
			features, labels=v16f.VGG16_featurize(imgfile)

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
			
			# features, labels=imf.image_featurize(cur_dir, haar_dir, imgfile)
			# features, labels=v16f.VGG16_featurize(imgfile)
			# features, labels=incf.inception_featurize(imgfile)
			# features, labels=xf.xception_featurize(imgfile)
			# features, labels=rf.resnet_featurize(imgfile)
			features, labels=v16f.VGG16_featurize(imgfile)

			# print(features)

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