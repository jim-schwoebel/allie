'''
Create a random set of features. 
'''

import librosa_features as lf 
import helpers.transcribe as ts
import random, math, os, sys, json

def prev_dir(directory):
    g=directory.split('/')
    dir_=''
    for i in range(len(g)):
        if i != len(g)-1:
            if i==0:
                dir_=dir_+g[i]
            else:
                dir_=dir_+'/'+g[i]
    # print(dir_)
    return dir_

directory=os.getcwd()
prevdir=prev_dir(directory)
sys.path.append(prevdir+'/text_features')
import nltk_features as nf 

# get features 
librosa_features, librosa_labels=lf.librosa_featurize('test.wav', False)
transcript=ts.transcribe_sphinx('test.wav')
nltk_features, nltk_labels=nf.nltk_featurize(transcript)

# relate some features to each other
# engineer 10 random features by dividing them and making new labels 
mixed_features=list()
mixed_labels=list()
mixed_inds=list()

for i in range(5):
    while len(mixed_labels) < 100: 

        # get some random features from both text and audio 
        i1=random.randint(0,len(librosa_features)-1)
        label_1=librosa_labels[i1]
        feature_1=librosa_features[i1]
        i2=random.randint(0,len(nltk_features)-1)
        label_2=nltk_labels[i2]
        feature_2=nltk_features[i2]
        # make new feature from labels 
        mixed_feature=feature_2/feature_1
        if mixed_feature != 0.0 and math.isnan(mixed_feature) == False and math.isinf(abs(mixed_feature)) == False:
            # make new label 
            mixed_label=label_2+' (nltk) ' + '| / | '+label_1 + ' (librosa)'
            print(mixed_label)
            mixed_labels.append(mixed_label)
            print(mixed_feature)
            mixed_features.append(mixed_feature)
            mixed_inds.append([i2,i1])

    data={'labels': mixed_labels,
          'mixed_inds': mixed_inds,
          'first_ind': 'nltk_features',
          'second_ind':'librosa_features'}

    jsonfile=open('mixed_feature_%s.json'%(str(i)),'w')
    json.dump(data,jsonfile)
    jsonfile.close() 

