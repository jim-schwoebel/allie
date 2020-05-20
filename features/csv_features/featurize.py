import os, json, wget, sys
import csv_features as cf
import os, wget, zipfile 
import shutil
import numpy as np
from tqdm import tqdm

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

def transcribe_csv(csv_file, csv_transcriber):
	if csv_transcriber=='raw text':
		transcript=open(csv_file).read()
	else:
		transcript=''
	return transcript 

def csv_featurize(feature_set, csvfile, cur_dir):

	if feature_set == 'csv_features':
		features, labels = cf.csv_featurize(csv_file, cur_dir)
		# make sure all the features do not have any infinity or NaN
		features=np.nan_to_num(np.array(features))
		features=features.tolist()

	else:
		print('-----------------------')
		print('!!		error		!!')
		print('-----------------------')
		print('Feature set %s does not exist. Please reformat the desired featurizer properly in the settings.json.'%(feature_set))
		print('Note that the featurizers are named accordingly with the Python scripts. csv_features.py --> csv_features in settings.json)')
		print('-----------------------')

	return features, labels 

##################################################
##				   Main script  		    	##
##################################################

basedir=os.getcwd()
help_dir=basedir+'/helpers'
prevdir=prev_dir(basedir)

foldername=sys.argv[1]
os.chdir(foldername)

# get class label from folder name 
labelname=foldername.split('/')
if labelname[-1]=='':
	labelname=labelname[-2]
else:
	labelname=labelname[-1]

listdir=os.listdir()
cur_dir=os.getcwd()

# settings 
g=json.load(open(prev_dir(prevdir)+'/settings.json'))
csv_transcribe=g['transcribe_csv']
default_csv_transcriber=g['default_csv_transcriber']

try:
	# assume 1 type of feature_set 
	feature_sets=[sys.argv[2]]
except:
	# if none provided in command line, then load deafult features 
	feature_sets=g['default_csv_features']

###################################################

# featurize all files accoridng to librosa featurize
for i in tqdm(range(len(listdir)), desc=labelname):

	# make audio file into spectrogram and analyze those images if audio file
	if listdir[i][-4:] in ['.csv']:
		try:
			csv_file=listdir[i]
			sampletype='csv'
			# I think it's okay to assume audio less than a minute here...
			if listdir[i][0:-4]+'.json' not in listdir:
				# make new .JSON if it is not there with base array schema.
				basearray=make_features(sampletype)

				# get the csv transcript  
				if csv_transcribe==True:
					for j in range(len(default_csv_transcriber)):
						csv_transcriber=default_csv_transcriber[j]
						transcript = transcribe_csv(csv_file, csv_transcriber)
						transcript_list=basearray['transcripts']
						transcript_list['csv'][csv_transcriber]=transcript 
						basearray['transcripts']=transcript_list

				# featurize the csv file 
				for j in range(len(feature_sets)):
					feature_set=feature_sets[j]
					features, labels = csv_featurize(feature_set, csv_file, cur_dir)
					try:
						data={'features':features.tolist(),
							  'labels': labels}
					except:
						data={'features':features,
							  'labels': labels}

					print(features)
					csv_features=basearray['features']['csv']
					csv_features[feature_set]=data
					basearray['features']['csv']=csv_features

				basearray['labels']=[labelname]

				# write to .JSON 
				jsonfile=open(listdir[i][0:-4]+'.json','w')
				json.dump(basearray, jsonfile)
				jsonfile.close()

			elif listdir[i][0:-4]+'.json' in listdir:
				# load the .JSON file if it is there 
				basearray=json.load(open(listdir[i][0:-4]+'.json'))
				transcript_list=basearray['transcripts']

				# only transcribe if you need to (checks within schema)
				if csv_transcribe==True: 
					for j in range(len(default_csv_transcriber)):
						csv_transcriber=default_csv_transcriber[j]
						if csv_transcriber not in list(transcript_list['csv']):
							transcript = transcribe_csv(csv_file, csv_transcriber)
							transcript_list['csv'][csv_transcriber]=transcript 
							basearray['transcripts']=transcript_list
						else:
							transcript = transcript_list['csv'][csv_transcriber]

				# only re-featurize if necessary (checks if relevant feature embedding exists)
				for j in range(len(feature_sets)):
					feature_set=feature_sets[j]
					if feature_set not in list(basearray['features']['csv']):
						features, labels = cf.csv_featurize(csv_file, cur_dir)
						print(features)

						try:
							data={'features':features.tolist(),
								  'labels': labels}
						except:
							data={'features':features,
								  'labels': labels}
						
						basearray['features']['csv'][feature_set]=data

				# only add the label if necessary 
				label_list=basearray['labels']
				if labelname not in label_list:
					label_list.append(labelname)
				basearray['labels']=label_list
				transcript_list=basearray['transcripts']

				# overwrite .JSON 
				jsonfile=open(listdir[i][0:-4]+'.json','w')
				json.dump(basearray, jsonfile)
				jsonfile.close()

		except:
			print('error')
