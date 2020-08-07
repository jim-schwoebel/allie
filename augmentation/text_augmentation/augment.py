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
  


 / _ \                                 | |      | | (_)            
/ /_\ \_   _  __ _ _ __ ___   ___ _ __ | |_ __ _| |_ _  ___  _ __  
|  _  | | | |/ _` | '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \ 
| | | | |_| | (_| | | | | | |  __/ | | | || (_| | |_| | (_) | | | |
\_| |_/\__,_|\__, |_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|
			  __/ |                                                
			 |___/                                                 
  ___  ______ _____       _____         _   
 / _ \ | ___ \_   _|  _  |_   _|       | |  
/ /_\ \| |_/ / | |   (_)   | | _____  _| |_ 
|  _  ||  __/  | |         | |/ _ \ \/ / __|
| | | || |    _| |_   _    | |  __/>  <| |_ 
\_| |_/\_|    \___/  (_)   \_/\___/_/\_\\__|

This section of Allie's API augments folders of .TXT files using
the default_text_augmenters.

Usage: python3 augment.py [folder] [augment_type]

All augment_type options include:
["augment_textacy", "augment_summary"]

Read more @ https://github.com/jim-schwoebel/allie/tree/master/augmentation/text_augmentation
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

def text_augment(augmentation_set, textfile, basedir):

	# only load the relevant featuresets for featurization to save memory
	if augmentation_set=='augment_eda':
		augment_eda.augment_eda(textfile, basedir)
	elif augmentation_set=='augment_textacy':
		augment_textacy.augment_textacy(textfile,basedir)

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
default_image_transcribers=settings['default_text_transcriber']
try:
	# assume 1 type of feature_set 
	augmentation_sets=[sys.argv[2]]
except:
	# if none provided in command line, then load deafult features 
	augmentation_sets=settings['default_text_augmenters']

################################################
##          Import According to settings      ##
################################################

# only load the relevant featuresets for featurization to save memory
if 'augment_eda' in augmentation_sets:
	import augment_eda
if 'augment_textacy' in augmentation_sets:
	import augment_textacy

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
##                NOW AUGMENT!!               ##
################################################

listdir=os.listdir()
random.shuffle(listdir)

# featurize all files accoridng to librosa featurize
for i in tqdm(range(len(listdir)), desc=labelname):
	if listdir[i][-4:] in ['.txt']:
		filename=[listdir[i]]
		for j in range(len(augmentation_sets)):
			augmentation_set=augmentation_sets[j]
			for k in range(len(filename)):
				filename=text_augment(augmentation_set, filename[k], basedir)
