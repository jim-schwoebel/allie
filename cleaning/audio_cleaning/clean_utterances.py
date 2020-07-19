import sys, os, shutil, librosa, uuid
from pyvad import vad, trim, split
import matplotlib.pyplot as plt
import numpy as np

# make sure the right version of numba is installed
os.system('pip3 install numba==0.48')

def clean_utterances(audiofile):
    '''
    taken from https://github.com/F-Tag/python-vad/blob/master/example.ipynb
    '''
    show=False
    curdir=os.getcwd()
    data, fs = librosa.core.load(audiofile)
    time = np.linspace(0, len(data)/fs, len(data))
    vact = vad(data, fs, fs_vad = 16000, hop_length = 30, vad_mode=3)
    vact = list(vact)
    while len(time) > len(vact):
        vact.append(0.0)
    utterances=list()

    for i in range(len(vact)):
        try:
            if vact[i] != vact[i-1]:
                # voice shift 
                if vact[i] == 1:
                    start = i
                else:
                    # this means it is end 
                    end = i
                    utterances.append([start,end])
        except:
            pass
    
    print(utterances)
    vact=np.array(vact)

    for i in range(len(utterances)):
        trimmed = data[utterances[i][0]:utterances[i][1]]
        tempfile = str(uuid.uuid4())+'.wav'
        librosa.output.write_wav(tempfile, trimmed, fs)

    os.remove(audiofile)