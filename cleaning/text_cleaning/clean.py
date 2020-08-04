'''
Import all the featurization scripts and allow the user to customize what embedding that
they would like to use for modeling purposes.

AudioSet is the only embedding that is a little bit wierd, as it is normalized to the length
of each audio file. There are many ways around this issue (such as normalizing to the length 
of each second), however, I included all the original embeddings here in case the time series
information is useful to you.
'''

################################################
##              IMPORT STATEMENTS             ##
################################################
import json, os, sys, time, random, uuid
import numpy as np 
# import helpers.transcribe as ts
# import speech_recognition as sr
from tqdm import tqdm

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

################################################
##              Helper functions              ##
################################################

def text_clean(cleaning_set, textfile, basedir):

	# long conditional on all the types of features that can happen and featurizes accordingly.
	if cleaning_set == 'clean_summary':
		clean_summary.clean_summary(textfile)
	elif cleaning_set == 'clean_textacy':
		clean_textacy.clean_textacy(textfile)

################################################
##              Load main settings            ##
################################################

# directory=sys.argv[1]
basedir=os.getcwd()
settingsdir=prev_dir(basedir)
settingsdir=prev_dir(settingsdir)
settings=json.load(open(settingsdir+'/settings.json'))
os.chdir(basedir)

text_transcribe=settings['transcribe_text']
default_text_transcribers=settings['default_text_transcriber']
try:
	# assume 1 type of feature_set 
	cleaning_sets=[sys.argv[2]]
except:
	# if none provided in command line, then load deafult features 
	cleaning_sets=settings['default_text_cleaners']

################################################
##          Import According to settings      ##
################################################

# only load the relevant featuresets for featurization to save memory
if 'clean_summary' in cleaning_sets:
	import clean_summary
elif 'clean_textacy' in cleaning_sets:
	import clean_textacy

################################################
##          Get featurization folder          ##
################################################

foldername=sys.argv[1]
os.chdir(foldername)
listdir=os.listdir() 
random.shuffle(listdir)
cur_dir=os.getcwd()
help_dir=basedir+'/helpers/'

# get class label from folder name 
labelname=foldername.split('/')
if labelname[-1]=='':
	labelname=labelname[-2]
else:
	labelname=labelname[-1]

################################################
##        REMOVE JSON AND DUPLICATES          ##
################################################

deleted_files=list()

# rename files appropriately
for i in range(len(listdir)):
	os.rename(listdir[i],listdir[i].replace(' ',''))

# remove duplicates / json files
for i in tqdm(range(len(listdir)), desc=labelname):
	file=listdir[i]
	listdir2=os.listdir()
	#now sub-loop through all files in directory and remove duplicates 
	for j in range(len(listdir2)):
		try:
			if listdir2[j]==file:
				pass
			elif listdir2[j]=='.DS_Store':
				pass 
			else:
				if filecmp.cmp(file, listdir2[j])==True:
					print('removing duplicate: %s ____ %s'%(file,listdir2[j]))
					deleted_files.append(listdir2[j])
					os.remove(listdir2[j])
				else:
					pass
		except:
			pass 
			
print('deleted the files below')
print(deleted_files)

listdir=os.listdir() 
for i in tqdm(range(len(listdir))):
	# remove .JSON files
	if listdir[i].endswith('.json'):
		os.remove(listdir[i])

# now rename files with UUIDs
listdir=os.listdir()
for i in range(len(listdir)):
	file=listdir[i]
	os.rename(file, str(uuid.uuid4())+file[-4:])
	
################################################
##                NOW CLEAN!!                 ##
################################################

listdir=os.listdir()
random.shuffle(listdir)

# featurize all files accoridng to librosa featurize
for i in tqdm(range(len(listdir)), desc=labelname):
	if listdir[i][-4:] in ['.txt']:
		filename=listdir[i]
		for j in range(len(cleaning_sets)):
			cleaning_set=cleaning_sets[j]
			text_clean(cleaning_set, filename, basedir)