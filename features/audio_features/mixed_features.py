'''
Create a random set of features. 
'''

import librosa_features as lf 
import helpers.transcribe as ts
import random, math, os, sys, json

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

def mixed_featurize(wavfile, transcript):
    print(os.getcwd())
    g=json.load(open(prev_dir(os.getcwd())+'/helpers/mixed/mixed_feature_0.json'))
    
    labels=g['labels']
    inds=g['mixed_inds']

    # get features 
    librosa_features, librosa_labels=lf.librosa_featurize(wavfile, False)
    nltk_features, nltk_labels=nf.nltk_featurize(transcript)
    features=list()

    for j in range(len(inds)):
        nltk_feature=inds[j][0]
        librosa_feature=inds[j][1]
        try:
            feature=nltk_feature/librosa_feature 
        except:
            # put zero value if the feature is not available 
            feature=0
        features.append(feature)

    return features, labels 
