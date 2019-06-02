'''
Import all the featurization scripts and allow the user to customize what embedding that
they would like to use for modeling purposes.

AudioSet is the only embedding that is a little bit wierd, as it is normalized to the length
of each audio file. There are many ways around this issue (such as normalizing to the length 
of each second), however, I included all the original embeddings here in case the time series
information is useful to you.
'''

import librosa_features as lf 
import standard_features as sf 
import audioset_features as af 
import sox_features as soxf 
import pyaudio_features as pf 
import sa_features as saf
import json, os, sys


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

# directory=sys.argv[1]
basedir=os.getcwd()
foldername=input('what is the name of the folder?')
os.chdir(foldername)
listdir=os.listdir() 

# feature_set='librosa_features'
# feature_set='standard_features'
# feature_set='audioset_features'
# feature_set='sox_features'
# feature_set='sa_features'
feature_set='pyaudio_features'


# featurize all files accoridng to librosa featurize
for i in range(len(listdir)):
	if listdir[i][-4:] in ['.wav', '.mp3']:
		#try:

		# I think it's okay to assume audio less than a minute here...

		# features, labels = lf.librosa_featurize(listdir[i], False)
		# features, labels = sf.standard_featurize(listdir[i])
		# features, labels = af.audioset_featurize(listdir[i], basedir, foldername)
		# features, labels = soxf.sox_featurize(listdir[i])
		# features, labels = saf.sa_featurize(listdir[i])
		
		features, labels = pf.pyaudio_featurize(listdir[i], basedir)
		print(features)

		try:
			data={'features':features.tolist(),
				  'labels': labels}
		except:
			data={'features':features,
				  'labels': labels}

		if listdir[i][0:-4]+'.json' not in listdir:
			# make new .JSON if it is not there with base array schema.
			basearray=make_features()
			audio_features=basearray['features']['audio']
			audio_features[feature_set]=data
			basearray['features']['audio']=audio_features
			basearray['labels']=[foldername]
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()
		elif listdir[i][0:-4]+'.json' in listdir:
			# overwrite existing .JSON if it is there.
			basearray=json.load(open(listdir[i][0:-4]+'.json'))
			basearray['features']['audio'][feature_set]=data
			basearray['labels']=[foldername]
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			json.dump(basearray, jsonfile)
			jsonfile.close()

		#except:
			#print('error')





