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


|  ___|       | |                        / _ \ | ___ \_   _|  _ 
| |_ ___  __ _| |_ _   _ _ __ ___  ___  / /_\ \| |_/ / | |   (_)
|  _/ _ \/ _` | __| | | | '__/ _ \/ __| |  _  ||  __/  | |      
| ||  __/ (_| | |_| |_| | | |  __/\__ \ | | | || |    _| |_   _ 
\_| \___|\__,_|\__|\__,_|_|  \___||___/ \_| |_/\_|    \___/  (_)
                                                                
                                                                
 _____  _____  _   _ 
/  __ \/  ___|| | | |
| /  \/\ `--. | | | |
| |     `--. \| | | |
| \__/\/\__/ /\ \_/ /
 \____/\____/  \___/ 
 
Note that this script is not used for Allie version 1.0 and will be 
updated in future releases.
 
Usage: python3 featurize.py [folder] [featuretype]

All featuretype options include:
["csv_features_regression"]

Read more @ https://github.com/jim-schwoebel/allie/tree/master/features/csv_features
'''
import os, json, wget, sys
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

def transcribe_csv(csv_file, csv_transcriber):
	if csv_transcriber=='raw text':
		transcript=open(csv_file).read()
	else:
		transcript=''
	return transcript 

def csv_featurize(features_dir, feature_set, csvfile, cur_dir):

	if feature_set == 'csv_features_regression':
		os.chdir(features_dir)
		if len(csvfile.split('featurized')) == 2 or len(csvfile.split('predictions'))==2:
			pass
		else:
			os.system('python3 featurize_csv_regression.py --input %s --output %s --target %s'%(cur_dir+'/'+csvfile, cur_dir+'/'+'featurized_'+csvfile, 'target'))
	else:
		print('-----------------------')
		print('!!		error		!!')
		print('-----------------------')
		print('Feature set %s does not exist. Please reformat the desired featurizer properly in the settings.json.'%(feature_set))
		print('Note that the featurizers are named accordingly with the Python scripts. csv_features.py --> csv_features in settings.json)')
		print('-----------------------')

##################################################
##				   Main script  		    	##
##################################################

basedir=os.getcwd()
help_dir=basedir+'/helpers'
prevdir=prev_dir(basedir)
features_dir=basedir
sys.path.append(prevdir)
from standard_array import make_features

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

if 'csv_features_regression' in feature_sets:
	pass

###################################################

# featurize all files accoridng to librosa featurize
for i in tqdm(range(len(listdir)), desc=labelname):

	# make audio file into spectrogram and analyze those images if audio file
	if listdir[i][-4:] in ['.csv']:
		try:
			csv_file=listdir[i]
			sampletype='csv'
			# I think it's okay to assume audio less than a minute here...
			if 'featurized_'+listdir[i] not in listdir:
				# featurize the csv file 
				for j in range(len(feature_sets)):
					feature_set=feature_sets[j]
					csv_featurize(features_dir, feature_set, csv_file, cur_dir)
			elif 'featurized_'+listdir[i] in listdir:
				pass			
		except:
			print('error')
