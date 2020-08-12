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
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee  

 / _ \                                 | |      | | (_)            
/ /_\ \_   _  __ _ _ __ ___   ___ _ __ | |_ __ _| |_ _  ___  _ __  
|  _  | | | |/ _` | '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \ 
| | | | |_| | (_| | | | | | |  __/ | | | || (_| | |_| | (_) | | | |
\_| |_/\__,_|\__, |_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|
              __/ |                                                
             |___/                                                 
  ___  ______ _____       _____  _____  _   _ 
 / _ \ | ___ \_   _|  _  /  __ \/  ___|| | | |
/ /_\ \| |_/ / | |   (_) | /  \/\ `--. | | | |
|  _  ||  __/  | |       | |     `--. \| | | |
| | | || |    _| |_   _  | \__/\/\__/ /\ \_/ /
\_| |_/\_|    \___/  (_)  \____/\____/  \___/ 

Augment CSV files for classification problems using
CTGAN.                                        
'''
import os
try:
    from ctgan import CTGANSynthesizer
except:
    os.system('pip3 install ctgan==0.2.1')
    from ctgan import CTGANSynthesizer
import time, random
import pandas as pd
import numpy as np

def augment_ctgan_regression(csvfile):
    data=pd.read_csv(csvfile)
    ctgan = CTGANSynthesizer()
    ctgan.fit(data,epochs=10) #15
    percent_generated=1
    df_gen = ctgan.sample(int(len(data)*percent_generated))
    print('augmented with %s samples'%(str(len(df_gen))))
    print(df_gen)
    # now add both togrther to make new .CSV file
    newfile1='augmented_'+csvfile
    df_gen.to_csv(newfile1, index=0)
    # now combine augmented and regular dataset
    data2=pd.read_csv('augmented_'+csvfile)
    frames = [data, data2]
    result = pd.concat(frames)
    newfile2='augmented_combined_'+csvfile
    result.to_csv(newfile2, index=0)
    return [csvfile,newfile1,newfile2]