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


/  __ \ |                (_)              / _ \ | ___ \_   _|  _ 
| /  \/ | ___  __ _ _ __  _ _ __   __ _  / /_\ \| |_/ / | |   (_)
| |   | |/ _ \/ _` | '_ \| | '_ \ / _` | |  _  ||  __/  | |      
| \__/\ |  __/ (_| | | | | | | | | (_| | | | | || |    _| |_   _ 
 \____/_|\___|\__,_|_| |_|_|_| |_|\__, | \_| |_/\_|    \___/  (_)
                                   __/ |                         
                                  |___/                          
  ___            _ _       
 / _ \          | (_)      
/ /_\ \_   _  __| |_  ___  
|  _  | | | |/ _` | |/ _ \ 
| | | | |_| | (_| | | (_) |
\_| |_/\__,_|\__,_|_|\___/ 
                           

This script takes in a folder of audio files and extracts out many
unique utterances from the audio files. Therefore, you get 

1.wav --> many audio files with utterances named as UUIDs.

This is useful if you are looking to create a large dataset of voiced utterances.

This is enabled if the default_audio_cleaners=['clean_utterances']
'''
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
        files=list()
        for i in range(len(utterances)):
                trimmed = data[utterances[i][0]:utterances[i][1]]
                tempfile = str(uuid.uuid4())+'.wav'
                librosa.output.write_wav(tempfile, trimmed, fs)
                files.append(tempfile)

        os.remove(audiofile)
        return files