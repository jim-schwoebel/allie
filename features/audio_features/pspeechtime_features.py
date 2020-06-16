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
##            LIBROSA_FEATURES.PY             ##    
================================================ 

Extracts acoustic features using the LibROSA library;
saves them as mean, standard devaition, amx, min, and median
in different classes: onset, rhythm, spectral, and power categories.

Note this is quite a powerful audio feature set that can be used
for a variety of purposes. 
'''
import numpy as np 
from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import ssc
import scipy.io.wavfile as wav
import os 

# get labels for later 
def get_labels(vector, label, label2):
    sample_list=list()
    for i in range(len(vector)):
        sample_list.append(label+str(i+1)+'_'+label2)

    return sample_list

def pspeech_featurize(file):
    # convert if .mp3 to .wav or it will fail 
    convert=False 
    if file[-4:]=='.mp3':
        convert=True 
        os.system('ffmpeg -i %s %s'%(file, file[0:-4]+'.wav'))
        file = file[0:-4] +'.wav'

    (rate,sig) = wav.read(file)
    mfcc_feat = mfcc(sig,rate).flatten().tolist()
    # fbank_feat = logfbank(sig,rate).flatten()
    # ssc_feat= ssc(sig, rate).flatten()

    while len(mfcc_feat) < 25948:
        mfcc_feat.append(0)
        
    features=mfcc_feat
    one=get_labels(mfcc_feat, 'mfcc_', 'time_25ms')
    # two=get_labels(fbank_feat, 'fbank_', 'time_25ms')
    # three=get_labels(ssc_feat, 'ssc_', 'time_25ms')

    labels=one
    # labels=one+two+three
    if convert==True:
        os.remove(file)

    print(len(labels))
    
    return features, labels 

# features, labels= pspeech_featurize('2495.wav')
# print(features.shape)
# print(len(labels))
