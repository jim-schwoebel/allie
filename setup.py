'''
Custom setup script for MacOSX.
'''
import os, json

def brew_install(modules):
  for i in range(len(modules)):
    os.system('brew install %s'%(modules[i]))
    
# assumes Mac OSX for SoX and FFmpeg installations
brew_modules=['sox, ffmpeg']
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
    data = {"default_audio_features": "audio_features", 
            "default_text_features": "nltk_features", 
            "default_image_features": "image_features", 
            "default_video_features": "video_features", 
            "default_csv_features": "csv_features",
            "bias_discovery": True, 
            "transcribe_audio": True,  
            "default_audio_transcriber": "pocketsphinx", 
            "transcribe_text": True, 
            "default_text_transcriber": "raw text",
            "transcribe_image": True, 
            "default_image_transcriber": "tesseract",
            "transcribe_videos": True, 
            "default_video_transcriber": "tesseract (averaged over frames)",
            "transcribe_csv": True, 
            "default_csv_transcriber": "raw text",
            "default_training_script": "tpot", 
            "augment_data": True, 
            "visualize_data": True,  
            "create_YAML": True, 
            "model_compress": True}
    jsonfile=open('settings.json','w')
    json.dump(data,jsonfile)
    jsonfile.close()
