'''
Allie CLI

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

Use the links above to learn more about each of these APIs, respectively.

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
		 'test', 'visualize']
sampletypes=['audio', 'text', 'image', 'video', 'csv']
problemtypes = ['c','r']

# get all the options from the terminal
parser = OptionParser()
parser.add_option("--c", "--command", dest="command",
                  help="the target command (annotate API = 'annotate',\n"+
	                  "augmentation API = 'augment',\n"+
	                  "cleaning API = 'clean',\n"+
	                  "datasets API = 'data'\n"+
	                  "features API = 'features'\n"+
	                  "model prediction API = 'predict'\n"+
	                  "preprocessing API = 'transform'\n"+
	                  "model training API = 'train'\n"+
	                  "testing API = 'test'\n"+
	                  "visualize API = visualize)", metavar="command")
parser.add_option("--p", "--problemtype", dest="problemtype",
				  help="specify the problem type (-c classification or -r regression)", metavar="problemtype")
parser.add_option("--s", "--sampletype", dest="sampletype",
				  help="specify the type files that you'd like to operate on (e.g. audio, text, image, video, csv)", metavar="sampletype")
parser.add_option("--n", "--name", dest="common_name",
				  help="specify the common name for the model (e.g. 'gender' for a male/female problem)", metavar="common_name")
parser.add_option("--i", "--class", dest="class_", 
				  help="specify the class that you wish to annotate for (e.g. male)", metavar="class_")

# load directory
parser.add_option("--l", "--ldir", dest="ldir",
                  help="the directory full of files to make model predictions; if not here will default to ./load_dir", metavar="ldir")

# up to 2 directories listed here
parser.add_option("--t1", "--tdir1", dest="tdir1",
                  help="the directory in the ./train_dir that represent the folders of files that the transform API will operate upon (e.g. 'males')", metavar="tdir1")
parser.add_option("--t2", "--tdir2", dest="tdir2",
                  help="the directory in the ./train_dir that represent the folders of files that the transform API will operate upon (e.g. 'females')", metavar="tdir2")

# featurization, cleaning, and augmentation directories
parser.add_option("--d1", "--dir1", dest="dir1",
                  help="the target directory that contains sample files for the features API, augmentation API, and cleaning API.", metavar="dir1")
parser.add_option("--d2", "--dir2", dest="dir2",
                  help="the target directory that contains sample files for the features API, augmentation API, and cleaning API.", metavar="dir2")
parser.add_option("--d3", "--dir3", dest="dir3",
                  help="the target directory that contains sample files for the features API, augmentation API, and cleaning API.", metavar="dir3")
parser.add_option("--d4", "--dir4", dest="dir4",
                  help="the target directory that contains sample files for the features API, augmentation API, and cleaning API.", metavar="dir4")

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
	ldir=options.ldir
except:
	pass
try:
	adir=options.adir
except:
	pass
try:
	tdir1=options.tdir1
except:
	pass
try:
	tdir2=options.tdir2
except:
	pass
try:
	dir1=options.dir1
except:
	pass
try:
	dir2=options.dir2
except:
	pass
try:
	dir3=options.dir3
except:
	pass
try:
	dir4=options.dir4
except:
	pass

# now pursue relevant command passed
if str(command) != 'None' and command in commands:

	if command == 'annotate':
		# - Annotate API - https://github.com/jim-schwoebel/allie/tree/master/annotation
		if str(adir) != 'None' and sampletype in sampletypes and str(class_) != 'None' and problemtype in problemtypes:
			os.chdir(annotation_dir)
			os.system('python3 annotate.py -d %s -s %s -c %s -p %s'%(adir, sampletype, class_, problemtype))
		else:
			if str(adir) == 'None':
				print('ERROR - annotation directory (-adir) not specified in the CLI')
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
			if str(dir1) != 'None':
				os.system('python3 augment.py %s'%(dir1))
			elif str(dir2) != 'None':
				os.system('python3 augment.py %s'%(dir2))
			elif str(dir3) != 'None':
				os.system('python3 augment.py %s'%(dir3))
			elif str(dir4) != 'None':
				os.system('python3 augment.py %s'%(dir4))
		else:	
			print('ERROR - '+sample +' - not in list of possible sample types: %s'%(str(sampletypes)))

	elif command == 'clean':
		# - Cleaning API - https://github.com/jim-schwoebel/allie/tree/master/cleaning
		if sampletype in sampletypes:
			os.chdir(clean_dir+'/%s_cleaning'%(sampletype))
			if str(dir1) != 'None':
				os.system('python3 clean.py %s'%(dir1))
			elif str(dir2) != 'None':
				os.system('python3 clean.py %s'%(dir2))
			elif str(dir3) != 'None':
				os.system('python3 clean.py %s'%(dir3))
			elif str(dir4) != 'None':
				os.system('python3 clean.py %s'%(dir4))
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
			if str(dir1) != 'None':
				os.system('python3 featurize.py %s'%(dir1))
			elif str(dir2) != 'None':
				os.system('python3 featurize.py %s'%(dir2))
			elif str(dir3) != 'None':
				os.system('python3 featurize.py %s'%(dir3))
			elif str(dir4) != 'None':
				os.system('python3 featurize.py %s'%(dir4))
		else:
			print('ERROR - '+sample +' - not in list of possible sample types: %s'%(str(sampletypes)))

	elif command == 'predict':
		# - Model Prediction API - https://github.com/jim-schwoebel/allie/tree/master/models
		if str(ldir) == 'None':
			print('Making model predictions in ./load_dir because ldir was not specified...')
			os.chdir(loadmodel_dir)
			os.system('python3 load.py')
		else:
			print('Making model predictions in the directory speciied: %s'%(str(ldir)))
			os.chdir(loadmodel_dir)
			os.system('python3 load.py %s'%(ldir))

	elif cmmand == 'transform':
		# - Preprocessing API - https://github.com/jim-schwoebel/allie/tree/master/preprocessing
		os.chdir(preprocessing_dir)
		# get first folder 
		if sampletype in sampletypes and problemtype in problemtypes and str(common_name) != 'None' and str(tdir1) != 'None' and str(tdir2) != 'None': 
			# get second folder 
			os.system('python3 transform.py %s %s %s %s %s'%(sampletype, problemtype, common_name, tdir1, tdir2))
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
else:
	print('ERROR - %s is not a valid command in the Allie CLI. Please use one of these commands'%(str(command)))
	print('\n')
	for i in range(len(commands)):
		print(' - '+commands[i])
	print('\n\n')
