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
                                                                             
 _____                                           _   _     _            
/  __ \                                         | | | |   (_)           
| /  \/ ___  _ __ ___  _ __ ___   __ _ _ __   __| | | |    _ _ __   ___ 
| |    / _ \| '_ ` _ \| '_ ` _ \ / _` | '_ \ / _` | | |   | | '_ \ / _ \
| \__/\ (_) | | | | | | | | | | | (_| | | | | (_| | | |___| | | | |  __/
 \____/\___/|_| |_| |_|_| |_| |_|\__,_|_| |_|\__,_| \_____/_|_| |_|\___|
                                                                        
                                                                        
 _____      _             __               
|_   _|    | |           / _|              
  | | _ __ | |_ ___ _ __| |_ __ _  ___ ___ 
  | || '_ \| __/ _ \ '__|  _/ _` |/ __/ _ \
 _| || | | | ||  __/ |  | || (_| | (_|  __/
 \___/_| |_|\__\___|_|  |_| \__,_|\___\___|


This is a command-line interface for all of Allie's APIs.

This makes it easy-to-call many common APIs within Allie,
which include:

	- Annotate API - https://github.com/jim-schwoebel/allie/tree/master/annotation
	- Augmentation API - https://github.com/jim-schwoebel/allie/tree/master/augmentation
	- Cleaning API - https://github.com/jim-schwoebel/allie/tree/master/cleaning
	- Datasets API - https://github.com/jim-schwoebel/allie/tree/master/datasets
	- Features API - https://github.com/jim-schwoebel/allie/tree/master/features
	- Model Prediction API - https://github.com/jim-schwoebel/allie/tree/master/models
	- Preprocessing API - https://github.com/jim-schwoebel/allie/tree/master/preprocessing
	- Model Training API - https://github.com/jim-schwoebel/allie/tree/master/training
	- Test API - https://github.com/jim-schwoebel/allie/tree/master/tests
	- Visualize API - https://github.com/jim-schwoebel/allie/tree/master/visualize

All of these commands will ingest the default settings via 
the settings.json document, so be sure to set the right settings
when calling the API.

Usage: allie.py [options]

Options:
  -h, --help            show this help message and exit
  --c=command, --command=command
                        the target command (annotate API = 'annotate',
                        augmentation API = 'augment',  cleaning API = 'clean',
                        datasets API = 'data',  features API = 'features',
                        model prediction API = 'predict',  preprocessing API =
                        'transform',  model training API = 'train',  testing
                        API = 'test',  visualize API = 'visualize',
                        list/change default settings = 'settings')
  --p=problemtype, --problemtype=problemtype
                        specify the problem type ('c' = classification or 'r'
                        = regression)
  --s=sampletype, --sampletype=sampletype
                        specify the type files that you'd like to operate on
                        (e.g. 'audio', 'text', 'image', 'video', 'csv')
  --n=common_name, --name=common_name
                        specify the common name for the model (e.g. 'gender'
                        for a male/female problem)
  --i=class_, --class=class_
                        specify the class that you wish to annotate (e.g.
                        'male')
  --d=dir, --dir=dir    an array of the target directory (or directories) that
                        contains sample files for the annotation API,
                        prediction API, features API, augmentation API,
                        cleaning API, and preprocessing API (e.g.
                        '/Users/jim/desktop/allie/train_dir/teens/')
			
If you have any questions or would like to contribute to our community,
please reach out to Jim Schwoebel @ js@neurolex.co
'''
import os, shutil, time, json
from optparse import OptionParser
from tqdm import tqdm
from pyfiglet import Figlet

# helper function to render modules and functions
def render(text, f):
	print(f.renderText(text))

f=Figlet(font='doh')
render('Allie',f)
f=Figlet(font='doom')
render('Command Line Interface',f)

###############################################################
##                     INITIALIZATION                        ##
###############################################################

# initialize variables for the test 
prevdir=os.getcwd()
load_dir = prevdir+'/load_dir'
train_dir = prevdir + '/train_dir'
model_dir = prevdir+ '/training'
features_dir=prevdir+'/features'
loadmodel_dir = prevdir+'/models'
clean_dir=prevdir+'/cleaning/'
data_dir=prevdir+'/datasets'
augment_dir=prevdir+'/augmentation'
test_dir=prevdir+'/tests'
visualization_dir=prevdir+'/visualize'
preprocessing_dir=prevdir+'/preprocessing'
annotation_dir=prevdir+'/annotation'

# settings
settings=json.load(open(prevdir+'/settings.json'))
clean_data=settings['clean_data']
augment_data=settings['augment_data']

# transcript settings 
default_audio_transcript=settings['default_audio_transcriber']
default_image_transcript=settings['default_image_transcriber']
default_text_transcript=settings['default_text_transcriber']
default_video_transcript=settings['default_video_transcriber']
default_csv_transcript=settings['default_csv_transcriber']
transcribe_audio=settings['transcribe_audio']
transcribe_text=settings['transcribe_text']
transcribe_image=settings['transcribe_image']
transcribe_video=settings['transcribe_video'] 
transcribe_csv=settings['transcribe_csv']

# feature settings 
default_audio_features=settings['default_audio_features']
default_text_features=settings['default_text_features']
default_image_features=settings['default_image_features']
default_video_features=settings['default_video_features']
default_csv_features=settings['default_csv_features']

# cleaning settings
default_audio_cleaners=settings['default_audio_cleaners']
default_text_cleaners=settings['default_text_cleaners']
default_image_cleaners=settings['default_image_cleaners']
default_video_cleaners=settings['default_video_cleaners']
default_csv_cleaners=settings['default_csv_cleaners']

# augmentation settings 
default_audio_augmenters=settings['default_audio_augmenters']
default_text_augmenters=settings['default_text_augmenters']
default_image_augmenters=settings['default_image_augmenters']
default_video_augmenters=settings['default_video_augmenters']
default_csv_augmenters=settings['default_csv_augmenters']

# preprocessing settings
select_features=settings['select_features']
reduce_dimensions=settings['reduce_dimensions']
scale_features=settings['scale_features']
default_scaler=settings['default_scaler']
default_feature_selector=settings['default_feature_selector']
default_dimensionality_reducer=settings['default_dimensionality_reducer']
dimension_number=settings['dimension_number']
feature_number=settings['feature_number']

# other settings for raining scripts 
training_scripts=settings['default_training_script']
model_compress=settings['model_compress']

# directories 
audiodir=loadmodel_dir+'/audio_models'
textdir=loadmodel_dir+'/text_models'
imagedir=loadmodel_dir+'/image_models'
videodir=loadmodel_dir+'/video_models'
csvdir=loadmodel_dir+'/csv_models'

# for if/then statements later
commands=['annotate', 'augment', 'clean', 'data', 
		 'features', 'predict', 'transform', 'train',
		 'test', 'visualize', 'settings']
sampletypes=['audio', 'text', 'image', 'video', 'csv']
problemtypes = ['c','r']

# get all the options from the terminal
parser = OptionParser()
parser.add_option("--c", "--command", dest="command",
                  help="the target command (annotate API = 'annotate', \n"+
	                  "augmentation API = 'augment', \n"+
	                  "cleaning API = 'clean', \n"+
	                  "datasets API = 'data', \n"+
	                  "features API = 'features', \n"+
	                  "model prediction API = 'predict', \n"+
	                  "preprocessing API = 'transform', \n"+
	                  "model training API = 'train', \n"+
	                  "testing API = 'test', \n"+
	                  "visualize API = 'visualize', \n" +
	                  "list/change default settings = 'settings')", metavar="command")
parser.add_option("--p", "--problemtype", dest="problemtype",
				  help="specify the problem type ('c' = classification or 'r' = regression)", metavar="problemtype")
parser.add_option("--s", "--sampletype", dest="sampletype",
				  help="specify the type files that you'd like to operate on (e.g. 'audio', 'text', 'image', 'video', 'csv')", metavar="sampletype")
parser.add_option("--n", "--name", dest="common_name",
				  help="specify the common name for the model (e.g. 'gender' for a male/female problem)", metavar="common_name")
parser.add_option("--i", "--class", dest="class_", 
				  help="specify the class that you wish to annotate (e.g. 'male')", metavar="class_")

# preprocessing, featurization, cleaning, and augmentation API directories (as an appended list)
parser.add_option("--d", "--dir", dest="dir",
                  help="an array of the target directory (or directories) that contains sample files for the annotation API, prediction API, features API, augmentation API, cleaning API, and preprocessing API (e.g. '/Users/jim/desktop/allie/train_dir/teens/')", metavar="dir",
                  action='append')

# parse arguments
(options, args) = parser.parse_args()

# pull arguments from CLI
try:
	command= options.command.lower().replace(' ','')
except:
	pass
try:
	common_name = options.common_name.lower()
except:
	pass
try:
	sampletype = options.sampletype.lower()
except:
	pass
try:
	problemtype = options.problemtype.lower()
except:
	pass
try:
	class_ = options.class_.lower()
except:
	pass
try:
	directory=options.dir
except:
	pass

# now pursue relevant command passed
try:
	if str(command) != 'None' and command in commands:

		if command == 'annotate':
			# - Annotate API - https://github.com/jim-schwoebel/allie/tree/master/annotation
			if str(directory) != 'None' and sampletype in sampletypes and str(class_) != 'None' and problemtype in problemtypes:
				for i in range(len(directory)):
					os.chdir(annotation_dir)
					os.system('python3 annotate.py -d %s -s %s -c %s -p %s'%(directory[i], sampletype, class_, problemtype))
			else:
				if str(directory) == 'None':
					print('ERROR - annotation directory (-dir) not specified in the CLI')
				elif sampletype not in sampletypes:
					print('ERROR - sample type (%s) not in possible sample types (%s)'%(str(sampletype), str(sampletypes)))
				elif str(class_) == 'None':
					print('ERROR - annotation class not specified (-class)')
				elif problemtype not in problemtypes:
					print('ERROR - probelm type (%s) not in possible problem types (%s)'%(str(problemtype),str(problemtypes)))
		elif command == 'augment':
			# - Augmentation API - https://github.com/jim-schwoebel/allie/tree/master/augmentation
			if sampletype in sampletypes:
				os.chdir(augment_dir+'/%s_augmentation'%(sampletype))
				if str(directory) != 'None':
					for i in range(len(directory)):
						os.system('python3 augment.py %s'%(directory[i]))
			else:	
				print('ERROR - '+sample +' - not in list of possible sample types: %s'%(str(sampletypes)))

		elif command == 'clean':
			# - Cleaning API - https://github.com/jim-schwoebel/allie/tree/master/cleaning
			if sampletype in sampletypes:
				os.chdir(clean_dir+'/%s_cleaning'%(sampletype))
				if str(directory) != 'None':
					for i in range(len(directory)):
						os.system('python3 clean.py %s'%(directory[i]))
			else:
				print('ERROR - '+sample +' - not in list of possible sample types: %s'%(str(sampletypes)))

		elif command == 'data':
			# - Datasets API - https://github.com/jim-schwoebel/allie/tree/master/datasets
			os.chdir(data_dir+'/downloads')
			os.system('python3 download.py')
			
		elif command == 'features':
			# - Features API - https://github.com/jim-schwoebel/allie/tree/master/features
			if sampletype in sampletypes:
				os.chdir(features_dir+'/%s_features'%(sampletype))
				if str(directory) != 'None':
					for i in range(len(directory)):
						os.system('python3 featurize.py %s'%(directory[i]))
			else:
				print('ERROR - '+sample +' - not in list of possible sample types: %s'%(str(sampletypes)))

		elif command == 'predict':
			# - Model Prediction API - https://github.com/jim-schwoebel/allie/tree/master/models
			if str(directory) == 'None':
				print('Making model predictions in ./load_dir because ldir was not specified...')
				os.chdir(loadmodel_dir)
				os.system('python3 load.py')
			else:
				print('Making model predictions in the directory specified: %s'%(str(ldir)))
				if str(directory) == 'None' and len(directory) == 1:
					os.chdir(loadmodel_dir)
					os.system('python3 load.py %s'%(directory[0]))
				else:
					print('too many directories (%s) specified for model prediction. \n\nPlease only specify one directory.'%(str(len(directory))))

		elif command == 'transform':
			# - Preprocessing API - https://github.com/jim-schwoebel/allie/tree/master/preprocessing
			os.chdir(preprocessing_dir)
			# get first folder 
			if sampletype in sampletypes and problemtype in problemtypes and str(common_name) != 'None' and str(directory) != 'None' and len(directory) > 1: 
				# get to N number of folders
				command='python3 transform.py %s %s %s'%(sampletype, problemtype, common_name)
				for i in range(len(directory)):
					command=command+' '+directory[i]
				os.system(command)
				print('your transform can now be found in the ./preprocessing/%s_transforms directory'%(sampletype))
			else:
				if str(tdir1) == 'None' or str(tdir2) == 'None':
					print('ERROR - transform API cannot be called. Please be sure that you defined your sample, problem type, common_name, and 2 directoreis properly (-tdir1 and -tdir2).')
				elif sampletype not in sampletypes:
					print('ERROR - '+sampletype +' not in possible sample types (%s)'%(str(sampletypes)))
				elif problem not in problemtypes:
					print('ERROR - '+problemtype + ' not in possible problem types (%s)'%(str(problemtypes)))
				elif str(common_name) == 'None':
					print('ERROR - common name not specified during creation of the transform in the preprocessing API.')

		elif command == 'train':
			# - https://github.com/jim-schwoebel/allie/tree/master/training
			os.chdir(model_dir)
			os.system('python3 model.py')
		elif command == 'test':
			# - Test API - https://github.com/jim-schwoebel/allie/tree/master/tests
			os.chdir(test_dir)
			os.system('python3 test.py')
		elif command == 'visualize':
			# - Visualize API - https://github.com/jim-schwoebel/allie/tree/master/visualize
			os.chdir(visualization_dir)
			os.system('python3 visualize.py')
		elif command == 'settings':
			print(settings)
			print('\n')
			settingslist=list(settings)
			textinput=input('Would you like to change any of these settings? Yes (-y) or No (-n)\n')
			if textinput.lower().replace(' ','') in ['yes','y']:
				textinput=input('What setting would you like to change?\n')
				while textinput not in settingslist and textinput.lower() != 'version':
					print('Setting not recognized, options are:')
					time.sleep(0.5)
					for i in range(len(settingslist)):
						if settingslist[i] != 'version':
							print('- '+settingslist[i])
							time.sleep(0.05)
					textinput=input('What setting would you like to change?\n')

				newsetting=input('What setting would you like to set here?\n')

				if str(newsetting).title() in ['True']:
					newsetting=True 
				elif str(newsetting).title() in ['False']:
					newsetting=False
				elif textinput in ['dimension_number', 'feature_number']:
					newsetting=int(newsetting)
				elif textinput in ['test_size']:
					newsetting=float(newsetting)
				else:
					settingnum=input('how many more settings would you like to set here?\n')
					newsetting=[newsetting]
					try:
						for i in range(int(settingnum)):
							newsetting2=input('What additional setting would you like to set here?\n')
							newsetting.append(newsetting2)
					except:
						pass

				print(type(newsetting))
				jsonfile=open('settings.json','w')
				settings[textinput]=newsetting
				json.dump(settings,jsonfile)
				jsonfile.close()

	else:	
		print('ERROR - %s is not a valid command in the Allie CLI. Please use one of these commands'%(str(command)))
		print('\n')
		for i in range(len(commands)):
			print(' - '+commands[i])
		print('\n\n')
except:
		print('ERROR - no command provided in the Allie CLI. \n\nPlease use one of these commands. \n')
		for i in range(len(commands)):
			print(' - '+commands[i])
		print('\n')
		print('Sample usage: \npython3 allie.py --command features --dir /Users/jimschwoebel/desktop/allie/train_dir/females --sampletype audio')
		print('\nFor additional help, type in:')
		print('python3 allie.py -h\n')
