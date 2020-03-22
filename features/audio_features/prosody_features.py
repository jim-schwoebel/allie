import argparse, json, os, sys
sys.path.append(os.getcwd()+'/helpers/DigiPsych_Prosody')
from helpers.DigiPsych_Prosody.prosody import Voice_Prosody
import pandas as pd
from datetime import datetime

'''
Featurize Wrapper for grabbing prosody features for audio stored in a folder
'''

def featurize_audio(audiofile,fsize):
    df = pd.DataFrame()
    vp = Voice_Prosody()
    if audiofile.endswith('.wav'):
        print('Prosody featurizing:',audiofile)
        feat_dict = vp.featurize_audio(audiofile,int(fsize))
        features=list(feat_dict.values())[0:-1]
        labels=list(feat_dict)[0:-1]
    
    print(features)
    print(labels)

    return features, labels

