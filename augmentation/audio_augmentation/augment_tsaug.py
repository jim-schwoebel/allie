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


 / _ \                                 | |      | | (_)            
/ /_\ \_   _  __ _ _ __ ___   ___ _ __ | |_ __ _| |_ _  ___  _ __  
|  _  | | | |/ _` | '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \ 
| | | | |_| | (_| | | | | | |  __/ | | | || (_| | |_| | (_) | | | |
\_| |_/\__,_|\__, |_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|
              __/ |                                                
             |___/                                                 
  ___  ______ _____        ___            _ _       
 / _ \ | ___ \_   _|  _   / _ \          | (_)      
/ /_\ \| |_/ / | |   (_) / /_\ \_   _  __| |_  ___  
|  _  ||  __/  | |       |  _  | | | |/ _` | |/ _ \ 
| | | || |    _| |_   _  | | | | |_| | (_| | | (_) |
\_| |_/\_|    \___/  (_) \_| |_/\__,_|\__,_|_|\___/   

Random crop subsequences of randomly spliced audio file.

with 50% probability, add random noise up to 1% - 5%,
drop out 10% of the time points (dropped out units are 1 ms, 
10 ms, or 100 ms) and fill the dropped out points with zeros.

More info @ https://tsaug.readthedocs.io/en/stable/
'''
import os, librosa
import numpy as np
import random
from tsaug import Crop, AddNoise, Dropout
import soundfile as sf

def augment_tsaug(filename):

        y, sr = librosa.load(filename, mono=False)
        duration=int(librosa.core.get_duration(y,sr))
        print(y.shape)
        # y=np.expand_dims(y.swapaxes(0,1), 0)

        # N second splice between 1 second to N-1 secondsd
        splice=random.randint(1,duration-1)

        my_augmenter = (Crop(size=sr * splice) * 5  # random crop subsequences of splice seconds
        + AddNoise(scale=(0.01, 0.05)) @ 0.5  # with 50% probability, add random noise up to 1% - 5%
        + Dropout(
                 p=0.1,
                 fill=0,
                 size=[int(0.001 * sr), int(0.01 * sr), int(0.1 * sr)]
                 )  # drop out 10% of the time points (dropped out units are 1 ms, 10 ms, or 100 ms) and fill the dropped out points with zeros
        )
        y_aug = my_augmenter.augment(y)
        newfile='tsaug_'+filename
        sf.write(newfile, y_aug.T, sr)
        return [filename, newfile]
