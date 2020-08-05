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
                           

This will featurize folders of audio files if the default_audio_features = ['pspeechtime_features']

This collects the time series features from python_speech_features.

See the documentation for more information: https://github.com/jameslyons/python_speech_features
'''
import numpy as np 
from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import ssc
import scipy.io.wavfile as wav
import os 

# get labels for later 
def get_labels(vector, label, label2):
    sample_list=list()
    for i in range(len(vector)):
        sample_list.append(label+str(i+1)+'_'+label2)

    return sample_list

def pspeech_featurize(file):
    # convert if .mp3 to .wav or it will fail 
    convert=False 
    if file[-4:]=='.mp3':
        convert=True 
        os.system('ffmpeg -i %s %s'%(file, file[0:-4]+'.wav'))
        file = file[0:-4] +'.wav'

    (rate,sig) = wav.read(file)
    mfcc_feat = mfcc(sig,rate).flatten().tolist()
    # fbank_feat = logfbank(sig,rate).flatten()
    # ssc_feat= ssc(sig, rate).flatten()

    while len(mfcc_feat) < 25948:
        mfcc_feat.append(0)
        
    features=mfcc_feat
    one=get_labels(mfcc_feat, 'mfcc_', 'time_25ms')
    # two=get_labels(fbank_feat, 'fbank_', 'time_25ms')
    # three=get_labels(ssc_feat, 'ssc_', 'time_25ms')

    labels=one
    # labels=one+two+three
    if convert==True:
        os.remove(file)

    print(len(labels))
    
    return features, labels 
