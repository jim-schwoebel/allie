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
        os.remove(audiofile)
        os.chdir(curdir)

        return features, labels