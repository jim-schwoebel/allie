import vgg16_features as v16f
import image_features as imf 
import inception_features as incf 
import xception_features as xf 
import resnet_features as rf 
import vgg19_features as v19f
import tesseract_features as tf
import helpers.audio_plot as ap 
import os, json

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

def transcribe(file, default_image_transcriber):

	# 3 types of transcripts = audio, text, and image 
	transcript_dict={'transcript': transcript,
					'transcript_type': 'image',
					'audio_transcriber': default_image_transcriber}

	return transcript_dict, transcript 

def image_featurize(feature_set, imgfile, cur_dir, haar_dir):

	if feature_set == 'image_features':
		features, labels=imf.image_featurize(cur_dir, haar_dir, imgfile)
	elif feature_set == 'vgg16_features':
		features, labels=v16f.vgg16_featurize(imgfile)
	elif feature_set == 'inception_features':
		features, labels=incf.inception_featurize(imgfile)
	elif feature_set == 'xception_features':
		features, labels=xf.xception_featurize(imgfile)
	elif feature_set == 'resnet_features':
		features, labels=rf.resnet_featurize(imgfile)
	elif feature_set == 'vgg19_features':
		features, labels=v16f.vgg19_featurize(imgfile)
	elif feature_set == 'tesseract_features':
		transcript, features, labels = tf.tesseract_featurize(imgfile)

	return features, labels 

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

##################################################
##				   Main script  		    	##
##################################################

# directory=sys.argv[1]
basedir=os.getcwd()
haar_dir=basedir+'/helpers/haarcascades'
foldername=sys.argv[1]
os.chdir(foldername)
cur_dir=os.getcwd()
listdir=os.listdir() 

# settings directory 
settingsdir=prev_dir(basedir)
settings=json.load(open(prev_dir(settingsdir)+'/settings.json'))
os.chdir(basedir)

image_transcribe=settings['transcribe_image']
default_image_transcriber=settings['default_image_transcriber']
feature_set=settings['default_image_features']

# get class label from folder name 
labelname=foldername.split('/')
if labelname[-1]=='':
	labelname=labelname[-2]
else:
	labelname=labelname[-1]

#### Can specify a few feature sets here (customizable in settings.json)
# feature_set='image_features'
# feature_set='VGG16_features'
# feature_set='Inception_features'
# feature_set='Xception_features'
# feature_set='Resnet50_features'
# feature_set='VGG19_features'
# feature_set='tesseract_features'

# featurize all files accoridng to librosa featurize
for i in range(len(listdir)):
	os.chdir(cur_dir)
	if listdir[i][-4:] in ['.jpg', '.png']:
		#try:
		imgfile=listdir[i]
		sampletype='image'

		if image_transcribe==True:
			transcript, features, labels = tf.tesseract_featurize(imgfile)
			transcript_dict, transcript =transcribe(transcript, default_image_transcriber)
		
		# get the features and labels according to specified feature_set 
		features, labels = image_featurize(feature_set, imgfile, cur_dir, haar_dir)

		# I think it's okay to assume audio less than a minute here...
		if listdir[i][0:-4]+'.json' not in listdir:
			# make new .JSON if it is not there with base array schema.
			basearray=make_features(sampletype)
			image_features=basearray['features']['image']
			basearray['transcript']=[transcript_dict]
			print(features)

			try:
				data={'features':features.tolist(),
					  'labels': labels}
			except:
				data={'features':features,
					  'labels': labels}

			image_features[feature_set]=data
			basearray['features']['image']=image_features
			basearray['labels']=[labelname]
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()
		elif listdir[i][0:-4]+'.json' in listdir:
			# overwrite existing .JSON if it is there.
			basearray=json.load(open(listdir[i][0:-4]+'.json'))
			transcript_list=basearray['transcript']
			transcript_list.append(transcript_dict)
			basearray['transcript']=transcript_list
			print(features)

			try:
				data={'features':features.tolist(),
					  'labels': labels}
			except:
				data={'features':features,
					  'labels': labels}

			basearray['features']['image'][feature_set]=data
			label_list=basearray['labels']
			if labelname not in label_list:
				label_list.append(labelname)
			basearray['labels']=label_list
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()

		#except:
			#print('error')
