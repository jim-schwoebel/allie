'''
Create a random set of features. 
'''

import librosa_features as lf 
import helpers.transcribe as ts
import random, math, os, sys, json
import numpy as np

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

directory=os.getcwd()
prevdir=prev_dir(directory)
sys.path.append(prevdir+'/text_features')
import nltk_features as nf 

def audiotext_featurize(wavfile, transcript):

    # get features 
    librosa_features, librosa_labels=lf.librosa_featurize(wavfile, False)
    nltk_features, nltk_labels=nf.nltk_featurize(transcript)

    features=np.append(np.array(librosa_features), np.array(nltk_features))
    labels=librosa_labels+nltk_labels 

    return features, labels 
