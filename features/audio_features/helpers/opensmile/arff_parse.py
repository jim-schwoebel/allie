import numpy as np
import json, os, time

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

# extract all features as different arff files
cmdlist=list()
feature_extractors=['avec2013.conf', 'emobase2010.conf', 'IS13_ComParE_Voc.conf', 'IS10_paraling.conf', 'IS13_ComParE.conf', 'IS10_paraling_compat.conf', 'emobase.conf', 
                    'emo_large.conf', 'IS11_speaker_state.conf', 'IS12_speaker_trait_compat.conf', 'IS09_emotion.conf', 'avec2011.conf', 'IS12_speaker_trait.conf', 
                    'prosodyShsViterbiLoudness.conf', 'ComParE_2016.conf', 'GeMAPSv01a.conf']

curdir=os.getcwd()
filedir=os.getcwd()+'/sample'

for i in range(len(feature_extractors)):
        feature_extractor=feature_extractors[i]
        arff_file=feature_extractor[0:-5]+'.arff'

        print(feature_extractor.upper())

        if feature_extractor== 'GeMAPSv01a.conf':
            os.system('SMILExtract -C opensmile-2.3.0/config/gemaps/%s -I sample/0.wav -O sample/%s'%(feature_extractor, arff_file))
        else:
            os.system('SMILExtract -C opensmile-2.3.0/config/%s -I sample/0.wav -O sample/%s'%(feature_extractor, arff_file))
        os.chdir(filedir)

        try:
            data, labels = parseArff(arff_file)
            print(len(data))
            print(len(labels))
            jsonfile=open(arff_file[0:-5]+'.json','w')
            data={'features': data,
                  'labels': labels}
            json.dump(data,jsonfile)
            jsonfile.close()
        except:
            print('error')
        os.chdir(curdir)

# print(labels)