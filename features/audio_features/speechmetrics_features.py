# the case of absolute metrics
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
                           

This will featurize folders of audio files if the default_audio_features = ['speechmetrics_features']

A wide array of open source audio quality measures to assess the quality of audio files. Note there 
are no audio file references necessary to extract these metrics.

taken from https://github.com/aliutkus/speechmetrics
'''
import os
# adding this in because some installations may not include speechmetrics
try:
    import speechmetrics
except:
    curdir=os.getcwd()
    os.system('pip3 install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics[cpu]')
    os.system('git clone https://github.com/jfsantos/SRMRpy')
    os.chdir('SRMRpy')
    os.system('python3 setup.py install')
    os.chdir(curdir)
    import speechmetrics
    
def speechmetrics_featurize(wavfile):
    window_length = 5 # seconds
    metrics = speechmetrics.load('absolute', window_length)
    scores = metrics(wavfile)
    scores['mosnet'] = float(scores['mosnet'])
    scores['srmr'] = float(scores['srmr'])
    features = list(scores.values())
    labels = list(scores)
    return features, labels
