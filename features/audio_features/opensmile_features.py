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
                           

This will featurize folders of audio files if the default_audio_features = ['opensmile_features']

Featurizes data with the OpenSMILE Toolkit: https://www.audeering.com/opensmile/

Note that this is a proprietary feature set and can be only used for research purposes.
Also note that you can specify a range of feature extractors within the script itself:

feature_extractors=['avec2013.conf', 'emobase2010.conf', 'IS10_paraling.conf', 'IS13_ComParE.conf', 'IS10_paraling_compat.conf', 'emobase.conf', 
                             'emo_large.conf', 'IS11_speaker_state.conf', 'IS12_speaker_trait_compat.conf', 'IS09_emotion.conf', 'IS12_speaker_trait.conf', 
                             'prosodyShsViterbiLoudness.conf', 'ComParE_2016.conf', 'GeMAPSv01a.conf']

The default setting is "GeMAPSv01a.conf" as this is the standard array used for vocal biomarker research studies.
'''
import numpy as np
import json, os, time, shutil

def parseArff(arff_file):
    '''
    Parses Arff File created by OpenSmile Feature Extraction
    '''
    f = open(arff_file,'r', encoding='utf-8')
    data = []
    labels = []
    for line in f:
        if '@attribute' in line:
            temp = line.split(" ")
            feature = temp[1]
            labels.append(feature)
        if ',' in line:
            temp = line.split(",")
            for item in temp:
                data.append(item)
    temp = arff_file.split('/')
    temp = temp[-1]
    data[0] = temp[:-5] + '.wav'

    newdata=list()
    newlabels=list()
    for i in range(len(data)):
        try:
            newdata.append(float(data[i]))
            newlabels.append(labels[i])
        except:
            pass
    return newdata,newlabels

def opensmile_featurize(audiofile, basedir, feature_extractor):

        # options 
        feature_extractors=['avec2013.conf', 'emobase2010.conf', 'IS10_paraling.conf', 'IS13_ComParE.conf', 'IS10_paraling_compat.conf', 'emobase.conf', 
                             'emo_large.conf', 'IS11_speaker_state.conf', 'IS12_speaker_trait_compat.conf', 'IS09_emotion.conf', 'IS12_speaker_trait.conf', 
                             'prosodyShsViterbiLoudness.conf', 'ComParE_2016.conf', 'GeMAPSv01a.conf']

        os.rename(audiofile,audiofile.replace(' ','_'))
        audiofile=audiofile.replace(' ','_')
        arff_file=audiofile[0:-4]+'.arff'
        curdir=os.getcwd()
        opensmile_folder=basedir+'/helpers/opensmile/opensmile-2.3.0'
        print(opensmile_folder)
        print(feature_extractor)
        print(audiofile)
        print(arff_file)

        if feature_extractor== 'GeMAPSv01a.conf':
            command='SMILExtract -C %s/config/gemaps/%s -I %s -O %s'%(opensmile_folder, feature_extractor, audiofile, arff_file)
            print(command)
            os.system(command)
        else:
            os.system('SMILExtract -C %s/config/%s -I %s -O %s'%(opensmile_folder, feature_extractor, audiofile, arff_file))

        features, labels = parseArff(arff_file)

        # remove temporary arff_file
        os.remove(arff_file)
        os.chdir(curdir)

        return features, labels
