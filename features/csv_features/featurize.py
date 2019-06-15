import os, json, wget, sys
import csv_features as cf
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

	transcripts={'audio': dict(),
				 'text': dict(),
				 'image': dict(),
				 'video': dict(),
				 'csv': dict()}

	data={'sampletype': sampletype,
		  'transcripts': transcripts,
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
foldername=sys.argv[1]
os.chdir(foldername)
cur_dir=os.getcwd()
listdir=os.listdir() 

# feature_set='video_features'
feature_set='y8m_features'

# get class label from folder name 
labelname=foldername.split('/')
if labelname[-1]=='':
	labelname=labelname[-2]
else:
	labelname=labelname[-1]

###################################################

# featurize all files accoridng to librosa featurize
for i in range(len(listdir)):

	# make audio file into spectrogram and analyze those images if audio file
	if listdir[i][-4:] in ['.csv']:
		#try:
		csvfile=listdir[i]
		sampletype='csv'
		# I think it's okay to assume audio less than a minute here...
		if listdir[i][0:-4]+'.json' not in listdir:
			# make new .JSON if it is not there with base array schema.
			basearray=make_features(sampletype)
			csv_features=basearray['features']['csv']
			features, labels, transcript = cf.csv_featurize(videofile, cur_dir, help_dir, fast_model)
			jsonfile=open(listdir[i][0:-4]+'.json','w')
			data={'featurized':'yes'}
			json.dump(data, jsonfile)
			jsonfile.close()

		elif listdir[i][0:-4]+'.json' in listdir:
			# pass if .JSON is here.
			pass 

		#except:
			#print('error')
