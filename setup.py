'''
Custom setup script for MacOSX.
'''
import os, json

def brew_install(modules):
  for i in range(len(modules)):
    os.system('brew install %s'%(modules[i]))
    
brew_modules=['sox']
brew_install(brew_modules)

# uninstall openCV to guarantee the right version...
os.system('pip3 uninstall opencv-python')
os.system('pip3 uninstall opencv-contrib-python')

# now install all modules with pip3 
os.system('pip3 install -r requirements.txt')

# install add-ons to NLTK 
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# install hyperopt-sklearn
curdir=os.getcwd()
os.chdir('training/helpers/hyperopt-sklearn')
os.system('pip3 install -e .')

# install keras-compressor
os.chdir(curdir)
os.chdir('training/keras-compressor')
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
    data = {"default_audio_features": "audio_features", 
            "default_text_features": "nltk_features", 
            "default_image_features": "image_features", 
            "default_video_features": "video_features", 
            "bias_discovery": true, 
            "transcribe_audio": true, 
            "default_audio_transcriber": "pocketsphinx", 
            "transcribe_image": true, 
            "default_image_transcriber": "tesseract",
            "transcribe_video": true, 
            "default_video_transcriber": "tesseract (averaged over frames)",
            "default_training_script": "keras", 
            "augment_data": true, 
            "visualize_data": true, 
            "create_YAML": true, 
            "model_compress": true}
    jsonfile=open('settings.json','w')
    json.dump(data,jsonfile)
    jsonfile.close()
