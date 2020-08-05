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
                           

Featurizes data with text feautures extracted from the transcript.

These text features include nltk_features, textacy_features, spacy_features, and text_features.

This will featurize folders of audio files if the default_audio_features = ['audiotext_features']
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
import textacy_features as tfe
import spacy_features as spf
import text_features as tfea

def audiotext_featurize(wavfile, transcript):

    # get features 
    # librosa_features, librosa_labels=lf.librosa_featurize(wavfile, False)
    nltk_features, nltk_labels=nf.nltk_featurize(transcript)
    textacy_features, textacy_labels=tfe.textacy_featurize(transcript)
    spacy_features, spacy_labels=spf.spacy_featurize(transcript)
    text_features,text_labels=tfea.text_featurize(transcript)

    # features=np.append(np.array(librosa_features), np.array(nltk_features))
    features=np.append(np.array(nltk_features),np.array(textacy_features))
    features=np.append(features,np.array(spacy_features))
    labels=nltk_labels+textacy_labels+spacy_labels
    # labels=librosa_labels+nltk_labels+textacy_labels+spacy_features

    return features, labels 
