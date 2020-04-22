import argparse, json, os, sys
sys.path.append(os.getcwd()+'/helpers/pyAudioLex')
from helpers.pyAudioLex import audio_ as audio
import pandas as pd
from datetime import datetime

'''
PyAudioLex features for the server.
'''

def pyaudiolex_featurize(filename):
    # features 
    results = audio.audio_featurize(filename)
    labels=list(results)
    features=list(results.values())

    # combine all features and values into proper format for Allie
    new_features=list()
    new_labels=list()
    for i in range(len(labels)):
        # print(labels[i])
        for j in range(len(features[i])):
            new_labels.append(labels[i]+'_window_'+str(j))
            new_features.append(features[i][j])

    features=new_features
    labels=new_labels

    # print(len(features))
    # print(len(labels))
    # print(labels)

    return features, labels
