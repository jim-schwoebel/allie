'''
Custom setup script for all operating systems.
'''
import os, json, sys

def brew_install(modules):
  for i in range(len(modules)):
    os.system('brew install %s'%(modules[i]))

# uninstall openCV to guarantee the right version
# this is the only step requiring manual intervention 
# so it's best to do this first.
os.system('pip3 install --upgrade pip')
os.system('pip3 uninstall opencv-python')
os.system('pip3 uninstall opencv-contrib-python')
os.system('pip3 install numpy')
curdir=os.getcwd()

# possible operating systems
# | Linux (2.x and 3.x) | linux2 (*) |
# | Windows             | win32      |
# | Windows/Cygwin      | cygwin     |
# | Windows/MSYS2       | msys       |
# | Mac OS X            | darwin     |
# | OS/2                | os2        |
# | OS/2 EMX            | os2emx     |

# assumes Mac OSX for SoX and FFmpeg installations
if sys.platform.lower() in ['darwin', 'os2', 'os2emx']:
  brew_modules=['sox', 'ffmpeg', 'opus-tools', 'opus', 'autoconf', 'automake', 'm4']
  brew_install(brew_modules)
  # install xcode if it is not already installed (on Mac) - important for OPENSMILE features
  os.system('xcode-select --install')
elif sys.platform.lower() in ['linux', 'linux2']:
  os.system('sudo apt-get install ffmpeg -y')
  os.system('sudo apt-get install sox -y')
  os.system('sudo apt-get install python-pyaudio -y')
  os.system('sudo apt-get install portaudio19-dev -y')
  os.system('sudo apt-get install libpq-dev python3.7-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev -y')
  os.system('sudo apt upgrade gcc -y')
  os.system('sudo apt-get install -y python python-dev python-pip build-essential swig git libpulse-dev')
  os.system('sudo apt-get install -y tesseract-ocr')
  os.system('sudo apt install -y opus-tools')
  os.system('sudo apt install -y libav-tools')
  os.system('sudo apt install -y libsm6')
elif sys.platform.lower() in ['win32', 'cygwin', 'msys']: 
  # https://www.thewindowsclub.com/how-to-install-ffmpeg-on-windows-10
  print('you have to install FFmpeg from source') 
  # https://github.com/JoFrhwld/FAVE/wiki/Sox-on-Windows
  print('you have to install SoX from source') 
  
# now install all modules with pip3 - install individually to reduce errors
requirements=open('requirements.txt').read().split('\n')
for i in range(len(requirements)):
	# skip pyobjc installations if not on mac computer
	if requirements[i].find('pyobjc')==0:
		if sys.platform.lower() in ['darwin', 'os2', 'os2emx']:
			os.system('pip3 install %s'%(requirements[i]))
		else:
			pass
	else:
		os.system('pip3 install %s'%(requirements[i]))

# install add-ons to NLTK 
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# install spacy add-ons
os.system('python3 -m spacy download en')
os.system("python3 -m spacy download 'en_core_web_sm'")

# install hyperopt-sklearn
curdir=os.getcwd()
os.chdir(curdir+'/training/helpers/hyperopt-sklearn')
os.system('pip3 install -e .')

# install keras-compressor
os.chdir(curdir)
os.chdir(curdir+'/training/keras_compressor')
os.system('pip3 install .')

# go back to host directory
os.chdir(curdir)

##################################################
##  ARCHIVED TRAINING REPOS (MAY ADD IN LATER)  ##
##################################################
# autokeras 
# autokeras-pretrained
# auto-sklearn

# create settings.json (if it does not exist)
listdir=os.listdir()
if 'settings.json' not in listdir:
    data = {'default_audio_features': ['standard_features'], 
            'default_text_features': ['nltk_features'], 
            'default_image_features': ['image_features'], 
            'default_video_features': ['video_features'], 
            'default_csv_features': ['csv_features'], 
            'bias_discovery': True, 
            'transcribe_audio': False, 
            'default_audio_transcriber': 'pocketsphinx', 
            'transcribe_text': True, 
            'default_text_transcriber': 'raw text', 
            'transcribe_image': True, 
            'default_image_transcriber': 'tesseract', 
            'transcribe_videos': True, 
            'default_video_transcriber': 'tesseract (averaged over frames)', 
            'transcribe_csv': True, 
            'default_csv_transcriber': 'raw text', 
            'default_training_script': ['tpot'], 
            'clean_data': False, 
            'augment_data': False, 
            'visualize_data': True, 
            'create_YAML': True, 
            'model_compress': False,
            'select_features': True, 
            'scale_features': True, 
            'reduce_dimensions': True, 
            'default_dimensionality_reducer': ['pca'], 
            'default_feature_selector': ['lasso'], 
            'default_scaler': ['standard_scaler']}
    jsonfile=open('settings.json','w')
    json.dump(data,jsonfile)
    jsonfile.close()

##################################################
##             AUTOMATED TESTING                ##
##################################################
'''
Perform some automated tests here to ensure add-on 
files are downloaded and features work properly.
'''
# testdir=curdir+'/tests'
# os.chdir(testdir)
# os.system('python3 test.py')
# os.chdir(curdir)
