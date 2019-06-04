'''
================================================ 
##            VOICEBOOK REPOSITORY            ##      
================================================ 

repository name: voicebook 
repository version: 1.0 
repository link: https://github.com/jim-schwoebel/voicebook 
author: Jim Schwoebel 
author contact: js@neurolex.co 
description: a book and repo to get you started programming voice applications in Python - 10 chapters and 200+ scripts. 
license category: opensource 
license: Apache 2.0 license 
organization name: NeuroLex Laboratories, Inc. 
location: Seattle, WA 
website: https://neurolex.ai 
release date: 2018-09-28 

This code (voicebook) is hereby released under a Apache 2.0 license license. 

For more information, check out the license terms below. 

================================================ 
##               LICENSE TERMS                ##      
================================================ 

Copyright 2018 NeuroLex Laboratories, Inc. 

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

     http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

================================================ 
##               SERVICE STATEMENT            ##        
================================================ 

If you are using the code written for a larger project, we are 
happy to consult with you and help you with deployment. Our team 
has >10 world experts in Kafka distributed architectures, microservices 
built on top of Node.js / Python / Docker, and applying machine learning to 
model speech and text data. 

We have helped a wide variety of enterprises - small businesses, 
researchers, enterprises, and/or independent developers. 

If you would like to work with us let us know @ js@neurolex.co. 

================================================ 
##            AUDIOSET_FEATURES.PY            ##    
================================================ 

Simple script to extract features using the VGGish model released by Google.

follows instructions from
https://github.com/tensorflow/models/tree/master/research/audioset

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

# define some initial helper functions 
def sync_record(filename, duration, fs, channels):
    print('recording')
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    sf.write(filename, myrecording, fs)
    print('done recording')

def setup_audioset(curdir):
    # Clone TensorFlow models repo into a 'models' directory.
    if 'models' in os.listdir():
        shutil.rmtree('models')
    os.system('git clone https://github.com/tensorflow/models.git')
    time.sleep(5)
    os.chdir(curdir+'/models/research/audioset')
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
    shutil.copytree(curdir+'/models/research/audioset/', curdir+'/audioset')
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
    labels='audioset_embedding'
    os.chdir(process_dir)

    return features, labels 

################################################################################
##                               MAIN SCRIPT                                  ##
################################################################################

# record a 10 second, mono 16k Hz audio file in the current directory
# filename='test.wav'
# sync_record(filename,10,16000,1)

# now let's featurize an audio sample in the current directory, test.wav 
# features, new_features =audioset_featurize(filename)
# print('new features')   
# print(new_features)
# print(len(new_features))




    
