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
                           

This will featurize folders of audio files if the default_audio_features = ['sa_features']

Uses the Signal Analysis library: https://brookemosby.github.io/Signal_Analysis/Signal_Analysis.features.html#module-Signal_Analysis.features.signal
'''
import os, librosa
try:
	from Signal_Analysis.features.signal import get_F_0, get_HNR, get_Jitter, get_Pulses
except:
	os.system('pip3 install Signal_Analysis==0.1.26')
	from Signal_Analysis.features.signal import get_F_0, get_HNR, get_Jitter, get_Pulses
import numpy as np

def sa_featurize(audiofile):
	'''
	from the docs 
	https://brookemosby.github.io/Signal_Analysis/Signal_Analysis.features.html#module-Signal_Analysis.features.signal
	'''

	y, sr = librosa.core.load(audiofile)
	duration = len(y)/sr
	print(duration)

	f0=get_F_0(y,sr)[0]
	hnr=get_HNR(y,sr)
	jitter=get_Jitter(y,sr)
	jitter_features=list(jitter.values())
	jitter_labels=list(jitter)
	pulses=get_Pulses(y,sr)
	pulses=len(pulses) / duration

	features=[f0,hnr,pulses]+jitter_features
	labels=['FundamentalFrequency','HarmonicstoNoiseRatio','PulsesPerSec']+jitter_labels

	print(dict(zip(labels,features)))

	return features, labels 
