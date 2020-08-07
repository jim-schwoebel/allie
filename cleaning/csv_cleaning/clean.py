'''
			   AAA               lllllll lllllll   iiii                      
			  A:::A              l:::::l l:::::l  i::::i                     
			 A:::::A             l:::::l l:::::l   iiii                      
			A:::::::A            l:::::l l:::::l                             
		   A:::::::::A            l::::l  l::::l iiiiiii     eeeeeeeeeeee    
		  A:::::A:::::A           l::::l  l::::l i:::::i   ee::::::::::::ee  
		 A:::::A A:::::A          l::::l  l::::l  i::::i  e::::::eeeee:::::ee
		A:::::A   A:::::A         l::::l  l::::l  i::::i e::::::e     e:::::e
	   A:::::A     A:::::A        l::::l  l::::l  i::::i e:::::::eeeee::::::e
	  A:::::AAAAAAAAA:::::A       l::::l  l::::l  i::::i e:::::::::::::::::e 
	 A:::::::::::::::::::::A      l::::l  l::::l  i::::i e::::::eeeeeeeeeee  
	A:::::AAAAAAAAAAAAA:::::A     l::::l  l::::l  i::::i e:::::::e           
   A:::::A             A:::::A   l::::::ll::::::li::::::ie::::::::e          
  A:::::A               A:::::A  l::::::ll::::::li::::::i e::::::::eeeeeeee  
 A:::::A                 A:::::A l::::::ll::::::li::::::i  ee:::::::::::::e  
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee  


/  __ \ |                (_)              / _ \ | ___ \_   _|  _ 
| /  \/ | ___  __ _ _ __  _ _ __   __ _  / /_\ \| |_/ / | |   (_)
| |   | |/ _ \/ _` | '_ \| | '_ \ / _` | |  _  ||  __/  | |      
| \__/\ |  __/ (_| | | | | | | | | (_| | | | | || |    _| |_   _ 
 \____/_|\___|\__,_|_| |_|_|_| |_|\__, | \_| |_/\_|    \___/  (_)
								   __/ |                         
								  |___/                          
 _____  _____  _   _ 
/  __ \/  ___|| | | |
| /  \/\ `--. | | | |
| |     `--. \| | | |
| \__/\/\__/ /\ \_/ /
 \____/\____/  \___/ 

This is Allie's Cleaning API for CSV Files.

Usage: python3 clean.py [folder] [cleantype]

All cleantype options include:
["clean_csv"]

Read more @ https://github.com/jim-schwoebel/allie/tree/master/cleaning/csv_cleaning
'''

################################################
##              IMPORT STATEMENTS             ##
################################################
import json, os, sys, time, random
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

def csv_clean(cleaning_set, csvfile, basedir):

	# long conditional on all the types of features that can happen and featurizes accordingly.
	if cleaning_set == 'clean_csv':
		clean_csv.clean_csv(csvfile, basedir)

################################################
##              Load main settings            ##
################################################

# directory=sys.argv[1]
basedir=os.getcwd()
settingsdir=prev_dir(basedir)
settingsdir=prev_dir(settingsdir)
settings=json.load(open(settingsdir+'/settings.json'))
os.chdir(basedir)

csv_transcribe=settings['transcribe_csv']
default_csv_transcribers=settings['default_csv_transcriber']
try:
	# assume 1 type of feature_set 
	cleaning_sets=[sys.argv[2]]
except:
	# if none provided in command line, then load deafult features 
	cleaning_sets=settings['default_csv_cleaners']

################################################
##          Import According to settings      ##
################################################

# only load the relevant featuresets for featurization to save memory
if 'clean_csv' in cleaning_sets:
	import clean_csv

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

################################################
##                NOW CLEAN!!                 ##
################################################

listdir=os.listdir()
random.shuffle(listdir)

# featurize all files accoridng to librosa featurize
for i in tqdm(range(len(listdir)), desc=labelname):
	if listdir[i][-4:] in ['.csv']:
		filename=listdir[i]
		for j in range(len(cleaning_sets)):
			cleaning_set=cleaning_sets[j]
			filename=csv_clean(cleaning_set, filename, basedir)