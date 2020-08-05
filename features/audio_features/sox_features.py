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
                           

This will featurize folders of audio files if the default_audio_features = ['sox_features']

Get features using SoX library, a workaround by outputting CLI in txt file and
uses a function to extract these features.
'''
import os
import numpy as np   

def clean_text(text):
    text=text.lower()
    chars=['a','b','c','d','e','f','g','h','i','j','k','l','m',
           'o','p','q','r','s','t','u','v','w','x','y','z',' ',
           ':', '(',')','-','=',"'.'"]
    for i in range(len(chars)):
        text=text.replace(chars[i],'')

    text=text.split('\n')
    new_text=list()
    # now get new text
    for i in range(len(text)):
        try:
            new_text.append(float(text[i].replace('\n','').replace('n','')))
        except:
            pass
            #print(text[i].replace('\n','').replace('n',''))
                        
    return new_text

def sox_featurize(filename):
    # soxi and stats files 
    soxifile=filename[0:-4].replace(' ','_')+'_soxi.txt'
    statfile=filename[0:-4].replace(' ','_')+'_stats.txt'
    if filename.endswith('.mp3'):
    	wavfile= filename[0:-4]+'.wav'
    	os.system('ffmpeg -i %s %s'%(filename,wavfile))
    	os.system('soxi %s > %s'%(wavfile, soxifile))
    	os.system('sox %s -n stat > %s 2>&1'%(wavfile, statfile))
    	os.remove(wavfile)
    else:
    	os.system('soxi %s > %s'%(filename, soxifile))
    	os.system('sox %s -n stat > %s 2>&1'%(filename, statfile))	
    # get basic info 
    s1=open(soxifile).read()
    s1_labels=['channels','samplerate','precision',
               'filesize','bitrate','sample encoding']
    s1=clean_text(s1)
    
    s2=open(statfile).read()
    s2_labels=['samples read','length','scaled by','maximum amplitude',
               'minimum amplitude','midline amplitude','mean norm','mean amplitude',
               'rms amplitude','max delta','min delta','mean delta',
               'rms delta','rough freq','vol adj']
    
    s2=clean_text(s2)

    labels=s1_labels+s2_labels
    features=np.array(s1+s2)
    os.remove(soxifile)
    os.remove(statfile)
 
    return features,labels

# features, labels = sox_featurize('test.wav')
