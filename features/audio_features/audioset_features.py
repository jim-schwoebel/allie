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
                                                                
                                                                
  ___            _ _       
 / _ \          | (_)      
/ /_\ \_   _  __| |_  ___  
|  _  | | | |/ _` | |/ _ \ 
| | | | |_| | (_| | | (_) |
\_| |_/\__,_|\__,_|_|\___/ 
                           

Simple script to extract features using the VGGish model released by Google.

follows instructions from
https://github.com/tensorflow/models/tree/master/research/audioset

This will featurize folders of audio files if the default_audio_features = ['audioset']

'''
################################################################################
##                         IMPORT STATEMENTS                                 ##
################################################################################

import os, shutil, json, time 
import sounddevice as sd
import soundfile as sf
import numpy as np
import tensorflow as tf

################################################################################
##                         HELPER FUNCTIONS                                  ##
################################################################################

def setup_audioset(curdir):
    # Clone TensorFlow models repo into a 'models' directory.
    if 'models' in os.listdir():
        shutil.rmtree('models')
    os.system('git clone https://github.com/tensorflow/models.git')
    time.sleep(5)
    os.chdir(curdir+'/models/research/audioset/vggish')
    # add modified file in the current folder 
    os.remove('vggish_inference_demo.py')
    shutil.copy(curdir+'/helpers/vggish_inference_demo.py', os.getcwd()+'/vggish_inference_demo.py')

    # Download data files into same directory as code.
    os.system('curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt')
    os.system('curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz')

    # Installation ready, let's test it.
    # If we see "Looks Good To Me", then we're all set.
    os.system('python3 vggish_smoke_test.py')

    # copy back into main directory and delete unnecessary models 
    shutil.copytree(curdir+'/models/research/audioset/vggish', curdir+'/audioset')
    shutil.rmtree(curdir+'/models')
    
    # go back to main directory
    os.chdir(curdir)
    
def audioset_featurize(filename, audioset_dir, process_dir):

    # get current directory 
    os.chdir(audioset_dir)
    curdir=os.getcwd()

    # download audioset files if audioset not in current directory 
    if 'audioset' not in os.listdir():
        #try:
        setup_audioset(curdir)
        #except:
            #print('there was an error installing audioset')

    # textfile definition to dump terminal outputs
    jsonfile=filename[0:-4]+'.json'
    # audioset folder
    curdir=os.getcwd()
    os.chdir(curdir+'/audioset')
    if 'processdir' not in os.listdir():
        os.mkdir('processdir')
    # need a .wav file here 
    if filename[-4:]=='.mp3':
        os.system('python3 vggish_inference_demo.py --mp3_file %s/%s'%(process_dir, filename))
    elif filename[-4:]=='.wav':
        os.system('python3 vggish_inference_demo.py --wav_file %s/%s'%(process_dir, filename))
    # now reference this .JSON file
    os.chdir(os.getcwd()+'/processdir')
    datafile=json.load(open(jsonfile))
    print(list(datafile))
    features=datafile['features']

    # output VGGish feature array and compressed means/stds 
    labels=list()
    for i in range(len(features)):
        labels.append('audioset_feature_%s'%(str(i+1)))
    os.chdir(process_dir)

    return features, labels 
