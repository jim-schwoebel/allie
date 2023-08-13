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
                           

This will featurize folders of audio files if the default_audio_features = ['pspeech_features']

Python Speech Features is a library for fast extraction of speech features like mfcc coefficients and 
log filter bank energies. Note that this library is much faster than LibROSA and other libraries, 
so it is useful to featurize very large datasets.

For more information, check out the documentation: https://github.com/jameslyons/python_speech_features
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
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)
    ssc_feat=ssc(sig, rate)

    one_=np.mean(mfcc_feat, axis=0)
    one=get_labels(one_, 'mfcc_', 'means')
    two_=np.std(mfcc_feat, axis=0)
    two=get_labels(one_, 'mfcc_', 'stds')
    three_=np.amax(mfcc_feat, axis=0)
    three=get_labels(one_, 'mfcc_', 'max')
    four_=np.amin(mfcc_feat, axis=0)
    four=get_labels(one_, 'mfcc_', 'min')
    five_=np.median(mfcc_feat, axis=0)
    five=get_labels(one_, 'mfcc_', 'medians')

    six_=np.mean(fbank_feat, axis=0)
    six=get_labels(six_, 'fbank_', 'means')
    seven_=np.std(fbank_feat, axis=0)
    seven=get_labels(six_, 'fbank_', 'stds')
    eight_=np.amax(fbank_feat, axis=0)
    eight=get_labels(six_, 'fbank_', 'max')
    nine_=np.amin(fbank_feat, axis=0)
    nine=get_labels(six_, 'fbank_', 'min')
    ten_=np.median(fbank_feat, axis=0)
    ten=get_labels(six_, 'fbank_', 'medians')

    eleven_=np.mean(ssc_feat, axis=0)
    eleven=get_labels(eleven_, 'spectral_centroid_', 'means')
    twelve_=np.std(ssc_feat, axis=0)
    twelve=get_labels(eleven_, 'spectral_centroid_', 'stds')
    thirteen_=np.amax(ssc_feat, axis=0)
    thirteen=get_labels(eleven_, 'spectral_centroid_', 'max')
    fourteen_=np.amin(ssc_feat, axis=0)
    fourteen=get_labels(eleven_, 'spectral_centroid_', 'min')
    fifteen_=np.median(ssc_feat, axis=0)
    fifteen=get_labels(eleven_, 'spectral_centroid_', 'medians')

    labels=one+two+three+four+five+six+seven+eight+nine+ten+eleven+twelve+thirteen+fourteen+fifteen
    features=np.append(one_,two_)
    features=np.append(features, three_)
    features=np.append(features, four_)
    features=np.append(features, five_)
    features=np.append(features, six_)
    features=np.append(features, seven_)
    features=np.append(features, eight_)
    features=np.append(features, nine_)
    features=np.append(features, ten_)
    features=np.append(features, eleven_)
    features=np.append(features, twelve_)
    features=np.append(features, thirteen_)
    features=np.append(features, fourteen_)
    features=np.append(features, fifteen_)

    if convert==True:
        os.remove(file)

    # print(features.shape)
    # print(len(labels))
    
    return features, labels 
