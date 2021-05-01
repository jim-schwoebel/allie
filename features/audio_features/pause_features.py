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

This will featurize folders of audio files if the default_audio_features = ['pause_features']

This extracts utterance times, pause numbers, time to first phonation,
and time to last phonation as features. 
'''

import sys, os
from pyvad import vad, trim, split
import librosa
import matplotlib.pyplot as plt
import numpy as np

# make sure the right version of numba is installed
os.system('pip3 install numba==0.48')

def pause_featurize(wavfile):
    '''
    taken from https://github.com/F-Tag/python-vad/blob/master/example.ipynb
    '''
    data, fs = librosa.core.load(wavfile)
    time = np.linspace(0, len(data)/fs, len(data))
    vact = vad(data, fs, fs_vad = 16000, hop_length = 30, vad_mode=2)
    vact = list(vact)
    while len(time) > len(vact):
        vact.append(0.0)
    utterances=list()

    for i in range(len(vact)):
        if vact[i] != vact[i-1]:
            # voice shift 
            if vact[i] == 1:
                start = i
            else:
                # this means it is end 
                end = i
                utterances.append([start/fs,end/fs])
    
    if len(utterances) > 0:
        first_phonation=utterances[0][0]
        last_phonation=utterances[len(utterances)-1][1]
    else:
        first_phonation=0
        last_phonation=0

    if len(utterences)-1 != -1:
        features = [utterances, len(utterances)-1, first_phonation, last_phonation]
    else:
        features = [utterances, 0, first_phonation, last_phonation]
        
    labels = ['UtteranceTimes', 'PauseNumber','TimeToFirstPhonation','TimeToLastPhonation']

    print(features)
    print(labels)
    
    return features, labels
