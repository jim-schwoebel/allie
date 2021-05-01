import sys, os
from pyvad import vad, trim, split
import librosa
import matplotlib.pyplot as plt
import numpy as np

# make sure the right version of numba is installed
os.system('pip3 install numba==0.48')

def pause_features(wavfile):
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
