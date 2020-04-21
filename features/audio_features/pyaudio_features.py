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
##             PYAUDIO_FEATURES.PY            ##    
================================================ 

Extract 170 pyaudioanalysis features
https://github.com/tyiannak/pyAudioAnalysis

Need python 2 and 3 installed because we use python2 version
of pyaudioanalysis
'''
import os,json, shutil
import numpy as np

def stats(matrix):
    mean=np.mean(matrix)
    std=np.std(matrix)
    maxv=np.amax(matrix)
    minv=np.amin(matrix)
    median=np.median(matrix)
    output=np.array([mean,std,maxv,minv,median])
    return output

def pyaudio_featurize(file, basedir):
    # use pyaudioanalysis library to export features
    # exported as file[0:-4].json 
    curdir=os.getcwd()
    shutil.copy(curdir+'/'+file, basedir+'/helpers/'+file)
    os.chdir(basedir+'/helpers/')
    os.system('python3 %s/helpers/pyaudio_help.py %s'%(basedir, file))
    jsonfile=file[0:-4]+'.json'
    g=json.load(open(jsonfile))
    features=g['features']
    labels=g['labels']
    os.remove(jsonfile)
    os.chdir(curdir)
    
    return features, labels
    
# features, labels =pyaudio_featurize('test.wav')
