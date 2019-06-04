'''
NLX-Textmodel microservice

Jim Schwoebel
(C) 2018, NeuroLex Laboratories 

This function takes in two folders with wav files (20 secs),
fingerprints them with audio features (.json),
and then builds an optimized machine learning model from these features.

This model-building scripts assumes the input audio files are of the same length.

The audio features tested include:
    Audio features across entire length 
    -mfcc 1 mean, min, max, std
    -mfcc 2 mean, min, max, std 
    -mfcc 3 mean, min, max, std 
    -mfcc 4 mean, min, max, std 
    -mfcc 5 mean, min, max, std
    -mfcc 6 mean, min, max, std
    -mfcc 7 mean, min, max, std
    -mfcc 8 mean, min, max, std
    -mfcc 9 mean, min, max, std 
    -mfcc 10 mean, min, max, std
    -mfcc 11 mean, min, max, std
    -mfcc 12 mean, min, max, std 
    -mfcc 13 mean, min, max, std

    Do each for 1 second interval (assuming 20 seconds)
    -[mfcc 1 mean, min, max, std] --> 0 to 1 sec
    -[mfcc 1 mean, min, max, std] --> 1 to 2 sec
    ...
    -[mfcc 1 mean, min, max, std] --> 18 to 19 sec
    -[mfcc 1 mean, min, max, std] --> 19 to 20 sec
    -[mfcc 2 mean, min, max, std] --> 0 to 1 sec
    -[mfcc 2 mean, min, max, std] --> 1 to 2 sec
    ...
    -[mfcc 2 mean, min, max, std] --> 18 to 19 sec
    -[mfcc 2 mean, min, max, std] --> 19 to 20 sec
    ...
    ...
    ...
    -[mfcc 13 mean, min, max, std] --> 18 to 19 sec
    -[mfcc 13 mean, min, max, std] --> 19 to 20 sec
    
    ~1040+52 features = ~1092 total
    
    -repeated words
    -... [future: embeddings from google)

#^ the features above are selected because it preserves some of the heirarchical nature of the data
(features over entire length and over time series). Also, there are many research papers that separate out
various groups using mfcc coefficients, well-known for dialect and gender detection. 

#^ note also that for audio files of varying length, we can average out the embeddings over 1 second period to
time series of N length, making the feature representation simpler. This could be instead reduced to a
104 feature embedding (see below)

    104 features (whole length) - numpy array A
    -mfcc 1 mean, min, max, std
    -mfcc 2 mean, min, max, std 
    -mfcc 3 mean, min, max, std 
    -mfcc 4 mean, min, max, std 
    -mfcc 5 mean, min, max, std
    -mfcc 6 mean, min, max, std
    -mfcc 7 mean, min, max, std
    -mfcc 8 mean, min, max, std
    -mfcc 9 mean, min, max, std 
    -mfcc 10 mean, min, max, std
    -mfcc 11 mean, min, max, std
    -mfcc 12 mean, min, max, std 
    -mfcc 13 mean, min, max, std
    -mfcc delta 1 mean, min, max, std
    -mfcc delta 2 mean, min, max, std 
    -mfcc delta 3 mean, min, max, std 
    -mfcc delta 4 mean, min, max, std 
    -mfcc delta 5 mean, min, max, std
    -mfcc delta 6 mean, min, max, std
    -mfcc delta 7 mean, min, max, std
    -mfcc delta 8 mean, min, max, std
    -mfcc delta 9 mean, min, max, std 
    -mfcc delta 10 mean, min, max, std
    -mfcc delta 11 mean, min, max, std
    -mfcc delta 12 mean, min, max, std 
    -mfcc delta 13 mean, min, max, std

    104 features - numpy array B [averaged]
    -[mfcc 1 mean(means), mean(mins), mean(maxs), mean(stds)] --> 0 to 1, 1 to 2, 2 to 3, ... 19 to 20.
    -[mfcc 2 mean(means), mean(mins), mean(maxs), mean(stds)] --> 0 to 1, 1 to 2, 2 to 3, ... 19 to 20.
    ...
    ...
    ...
    -[mfcc 13 mean(means), mean(mins), mean(maxs), mean(stds)] --> 0 to 1, 1 to 2, 2 to 3, ... 19 to 20.

    np.append(A,B) --> 104 features 

#^ it is this approach that we take in the current representation [208 features]

The models tested here include:
    -Naive Bayes
    -Decision tree 
    -Support vector machines
    -Bernoulli
    -Maximum entropy
    -Adaboost
    -Gradient boost
    -Logistic regression
    -Hard voting
    -K nearest neighbors
    -Random forest 
    -SVM algorithm
    -... [future: Deep learning models, etc.]

The output is an optimized machine learning model to a feature as a
.pickle file, which can be easily imported into the future through code like:

    import pickle
    f = open(classifiername+'_%s'%(selectedfeature)+'.pickle', 'rb')
    classifier = pickle.load(function(f))
    ##where function is the feature 
    f.close()
    ##classify with proper function...
    classifier.classify(startword(text))

Happy modeling!!
'''

import librosa
from pydub import AudioSegment
import os, nltk, random, json 
from nltk import word_tokenize 
from nltk.classify import apply_features, SklearnClassifier, maxent
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from textblob import TextBlob
from operator import itemgetter
import getpass
import numpy as np
import pickle
import datetime 
import time 

# INITIAL FUNCTIONS
#############################################################
def optimizemodel_sc(train_set2,labels_train_set2,test_set2,labels_test_set2,modelname,classes,testing_set,min_num,selectedfeature,training_data):
    filename=modelname
    start=time.time()
    jmsgs=train_set2+test_set2
    omsgs=labels_train_set2+labels_test_set2
    
    c1=0
    c5=0

    try:
        #decision tree
        classifier2 = DecisionTreeClassifier(random_state=0)
        classifier2.fit(train_set2,labels_train_set2)
        scores = cross_val_score(classifier2, test_set2, labels_test_set2,cv=5)
        print('Decision tree accuracy (+/-) %s'%(str(scores.std())))
        c2=scores.mean()
        c2s=scores.std()
        print(c2)
    except:
        c2=0
        c2s=0

    try:
        classifier3 = GaussianNB()
        classifier3.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier3, test_set2, labels_test_set2,cv=5)
        print('Gaussian NB accuracy (+/-) %s'%(str(scores.std())))
        c3=scores.mean()
        c3s=scores.std()
        print(c3)
    except:
        c3=0
        c3s=0

    try:
        #svc 
        classifier4 = SVC()
        classifier4.fit(train_set2,labels_train_set2)
        scores=cross_val_score(classifier4, test_set2, labels_test_set2,cv=5)
        print('SKlearn classifier accuracy (+/-) %s'%(str(scores.std())))
        c4=scores.mean()
        c4s=scores.std()
        print(c4)
    except:
        c4=0
        c4s=0

    try:
        #adaboost
        classifier6 = AdaBoostClassifier(n_estimators=100)
        classifier6.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier6, test_set2, labels_test_set2,cv=5)
        print('Adaboost classifier accuracy (+/-) %s'%(str(scores.std())))
        c6=scores.mean()
        c6s=scores.std()
        print(c6)
    except:
        c6=0
        c6s=0

    try:
        #gradient boosting 
        classifier7=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        classifier7.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier7, test_set2, labels_test_set2,cv=5)
        print('Gradient boosting accuracy (+/-) %s'%(str(scores.std())))
        c7=scores.mean()
        c7s=scores.std()
        print(c7)
    except:
        c7=0
        c7s=0

    try:
        #logistic regression
        classifier8=LogisticRegression(random_state=1)
        classifier8.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier8, test_set2, labels_test_set2,cv=5)
        print('Logistic regression accuracy (+/-) %s'%(str(scores.std())))
        c8=scores.mean()
        c8s=scores.std()
        print(c8)
    except:
        c8=0
        c8s=0

    try:
        #voting 
        classifier9=VotingClassifier(estimators=[('gradboost', classifier7), ('logit', classifier8), ('adaboost', classifier6)], voting='hard')
        classifier9.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier9, test_set2, labels_test_set2,cv=5)
        print('Hard voting accuracy (+/-) %s'%(str(scores.std())))
        c9=scores.mean()
        c9s=scores.std()
        print(c9)
    except:
        c9=0
        c9s=0

    try:
        #knn
        classifier10=KNeighborsClassifier(n_neighbors=7)
        classifier10.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier10, test_set2, labels_test_set2,cv=5)
        print('K Nearest Neighbors accuracy (+/-) %s'%(str(scores.std())))
        c10=scores.mean()
        c10s=scores.std()
        print(c10)
    except:
        c10=0
        c10s=0

    try:
        #randomforest
        classifier11=RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        classifier11.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier11, test_set2, labels_test_set2,cv=5)
        print('Random forest accuracy (+/-) %s'%(str(scores.std())))
        c11=scores.mean()
        c11s=scores.std()
        print(c11)
    except:
        c11=0
        c11s=0

    try:
##        #svm
        classifier12 = svm.SVC(kernel='linear', C = 1.0)
        classifier12.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier12, test_set2, labels_test_set2,cv=5)
        print('svm accuracy (+/-) %s'%(str(scores.std())))
        c12=scores.mean()
        c12s=scores.std()
        print(c12)
    except:
        c12=0
        c12s=0

    #IF IMBALANCED, USE http://scikit-learn.org/dev/modules/generated/sklearn.naive_bayes.ComplementNB.html

    maxacc=max([c2,c3,c4,c6,c7,c8,c9,c10,c11,c12])

    # if maxacc==c1:
    #     print('most accurate classifier is Naive Bayes'+'with %s'%(selectedfeature))
    #     classifiername='naive-bayes'
    #     classifier=classifier1
    #     #show most important features
    #     classifier1.show_most_informative_features(5)
    if maxacc==c2:
        print('most accurate classifier is Decision Tree'+'with %s'%(selectedfeature))
        classifiername='decision-tree'
        classifier2 = DecisionTreeClassifier(random_state=0)
        classifier2.fit(train_set2+test_set2,labels_train_set2+labels_test_set2)
        classifier=classifier2
    elif maxacc==c3:
        print('most accurate classifier is Gaussian NB'+'with %s'%(selectedfeature))
        classifiername='gaussian-nb'
        classifier3 = GaussianNB()
        classifier3.fit(train_set2+test_set2, labels_train_set2+labels_test_set2)
        classifier=classifier3
    elif maxacc==c4:
        print('most accurate classifier is SK Learn'+'with %s'%(selectedfeature))
        classifiername='sk'
        classifier4 = SVC()
        classifier4.fit(train_set2+test_set2,labels_train_set2+labels_test_set2)
        classifier=classifier4
    elif maxacc==c5:
        print('most accurate classifier is Maximum Entropy Classifier'+'with %s'%(selectedfeature))
        classifiername='max-entropy'
        classifier=classifier5
    #can stop here (c6-c10)
    elif maxacc==c6:
        print('most accuracate classifier is Adaboost classifier'+'with %s'%(selectedfeature))
        classifiername='adaboost'
        classifier6 = AdaBoostClassifier(n_estimators=100)
        classifier6.fit(train_set2+test_set2, labels_train_set2+labels_test_set2)
        classifier=classifier6
    elif maxacc==c7:
        print('most accurate classifier is Gradient Boosting '+'with %s'%(selectedfeature))
        classifiername='graidentboost'
        classifier7=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        classifier7.fit(train_set2+test_set2, labels_train_set2+labels_test_set2)
        classifier=classifier7
    elif maxacc==c8:
        print('most accurate classifier is Logistic Regression '+'with %s'%(selectedfeature))
        classifiername='logistic_regression'
        classifier8=LogisticRegression(random_state=1)
        classifier8.fit(train_set2+test_set2, labels_train_set2+labels_test_set2)
        classifier=classifier8
    elif maxacc==c9:
        print('most accurate classifier is Hard Voting '+'with %s'%(selectedfeature))
        classifiername='hardvoting'
        classifier7=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        classifier7.fit(train_set2+test_set2, labels_train_set2+labels_test_set2)
        classifier8=LogisticRegression(random_state=1)
        classifier8.fit(train_set2+test_set2, labels_train_set2+labels_test_set2)
        classifier6 = AdaBoostClassifier(n_estimators=100)
        classifier6.fit(train_set2+test_set2, labels_train_set2+labels_test_set2)
        classifier9=VotingClassifier(estimators=[('gradboost', classifier7), ('logit', classifier8), ('adaboost', classifier6)], voting='hard')
        classifier9.fit(train_set2+test_set2, labels_train_set2+labels_test_set2)
        classifier=classifier9
    elif maxacc==c10:
        print('most accurate classifier is K nearest neighbors '+'with %s'%(selectedfeature))
        classifiername='knn'
        classifier10=KNeighborsClassifier(n_neighbors=7)
        classifier10.fit(train_set2+test_set2, labels_train_set2+labels_test_set2)
        classifier=classifier10
    elif maxacc==c11:
        print('most accurate classifier is Random forest '+'with %s'%(selectedfeature))
        classifiername='randomforest'
        classifier11=RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        classifier11.fit(train_set2+test_set2, labels_train_set2+labels_test_set2)
        classifier=classifier11
    elif maxacc==c12:
        print('most accurate classifier is SVM '+' with %s'%(selectedfeature))
        classifiername='svm'
        classifier12 = svm.SVC(kernel='linear', C = 1.0)
        classifier12.fit(train_set2+test_set2, labels_train_set2+labels_test_set2)    
        classifier=classifier12

    modeltypes=['decision-tree','gaussian-nb','sk','adaboost','gradient boosting','logistic regression','hard voting','knn','random forest','svm']
    accuracym=[c2,c3,c4,c6,c7,c8,c9,c10,c11,c12]
    accuracys=[c2s,c3s,c4s,c6s,c7s,c8s,c9s,c10s,c11s,c12s]
    model_accuracy=list()
    for i in range(len(modeltypes)):
        model_accuracy.append([modeltypes[i],accuracym[i],accuracys[i]])

    model_accuracy.sort(key=itemgetter(1))
    endlen=len(model_accuracy)

    print('saving classifier to disk')
    f=open(modelname+'.pickle','wb')
    pickle.dump(classifier,f)
    f.close()

    end=time.time()

    execution=end-start
    
    print('summarizing session...')

    accstring=''
    
    for i in range(len(model_accuracy)):
        accstring=accstring+'%s: %s (+/- %s)\n'%(str(model_accuracy[i][0]),str(model_accuracy[i][1]),str(model_accuracy[i][2]))

    training=len(train_set2)
    testing=len(test_set2)
    
    summary='SUMMARY OF MODEL SELECTION \n\n'+'WINNING MODEL: \n\n'+'%s: %s (+/- %s) \n\n'%(str(model_accuracy[len(model_accuracy)-1][0]),str(model_accuracy[len(model_accuracy)-1][1]),str(model_accuracy[len(model_accuracy)-1][2]))+'MODEL FILE NAME: \n\n %s.pickle'%(filename)+'\n\n'+'DATE CREATED: \n\n %s'%(datetime.datetime.now())+'\n\n'+'EXECUTION TIME: \n\n %s\n\n'%(str(execution))+'GROUPS: \n\n'+str(classes)+'\n'+'('+str(min_num)+' in each class, '+str(int(testing_set*100))+'% used for testing)'+'\n\n'+'TRAINING SUMMARY:'+'\n\n'+training_data+'FEATURES: \n\n %s'%(selectedfeature)+'\n\n'+'MODELS, ACCURACIES, AND STANDARD DEVIATIONS: \n\n'+accstring+'\n\n'+'(C) 2018, NeuroLex Laboratories'

    data={
        'model':modelname,
        'modeltype':model_accuracy[len(model_accuracy)-1][0],
        'accuracy':model_accuracy[len(model_accuracy)-1][1],
        'deviation':model_accuracy[len(model_accuracy)-1][2]
        }
    
    return [classifier, model_accuracy[endlen-1], summary, data]

def statslist(veclist):
    newlist=list()
    #fingerprint statistical features
    #append each with mean, std, var, median, min, and max
    if len(veclist)>100:
        newlist=[float(np.mean(veclist)),float(np.std(veclist)),float(np.var(veclist)),
             float(np.median(veclist)),float(np.amin(veclist)),float(np.amax(veclist))]
        newlist2=newlist 
    else:
        for i in range(len(veclist)):
            newlist.append([float(np.mean(veclist[i])),float(np.std(veclist[i])),float(np.var(veclist[i])),
                  float(np.median(veclist[i])),float(np.amin(veclist[i])),float(np.amax(veclist[i]))])           
        newlist2=list()
        for i in range(len(newlist)):
            newlist2=newlist2+newlist[i]
    return newlist2

def audio_features(filename):

    hop_length = 512
    n_fft=2048

    #load file 
    y, sr = librosa.load(filename)
    duration=float(librosa.core.get_duration(y))
    #extract features from librosa 
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    y_harmonic,y_percussive=librosa.effects.hpss(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=1)
    mfcc_delta = librosa.feature.delta(mfcc)
    zero_crossings = librosa.zero_crossings(y)
    zero_crossing_time = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y)
    spectral_contrast = librosa.feature.spectral_contrast(y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y)
    rmse=librosa.feature.rmse(y)

    features=[tempo, len(beat_frames), beat_times[0], beat_times[len(beat_times)-1],
              np.mean(y_harmonic), np.std(y_harmonic), np.mean(y_percussive), np.std(y_percussive),
              np.mean(zero_crossing_time[0])]
    features=features+statslist(mfcc[0])+statslist(mfcc_delta[0])+statslist(spectral_centroid[0])+statslist(spectral_bandwidth[0])
    features=features+statslist(spectral_contrast[0])+statslist(spectral_rolloff[0])+statslist(rmse[0])

    features=np.array(features)

    return features

#FEATURIZE .WAV FILES WITH AUDIO FEATURES --> MAKE JSON (if needed)
#############################################################

classnum=input('how many classes are you training?')

folderlist=list()
a=0
while a != int(classnum):
    folderlist.append(input('what is the folder name for class %s?'%(str(a+1))))
    a=a+1

name=''
for i in range(len(folderlist)):
    if i==0:
        name=name+folderlist[i]
    else:
        name=name+'_'+folderlist[i]
    
start=time.time()
#modelname=input('what is the name of your classifier?')
modelname=name+'_sc_audio'

jsonfilename=name+'.json'
dir3=os.getcwd()+'/nlx-audiomodel/'
model_dir=os.getcwd()+'/nlx-audiomodel/models'
cur_dir=dir3
testing_set=0.25

try:
    os.chdir(dir3)
except:
    os.mkdir(dir3)
    os.chdir(dir3)

if jsonfilename not in os.listdir():

    features_list=list()
    
    for i in range(len(folderlist)):

        name=folderlist[i]

        dir_=cur_dir+name

        g='error'
        while g == 'error':
            try:
                g='noterror'
                os.chdir(dir_)
            except:
                g='error'
                print('directory not recognized')
                dir_=input('input directory %s path'%(str(i+1)))


        #now go through each directory and featurize the samples and save them as .json files 
        try:
            os.chdir(dir_)
        except:
            os.mkdir(dir_)
            os.chdir(dir_)
            
        dirlist=os.listdir()

        #if broken session, load all previous transcripts
        #this reduces costs if tied to GCP
        one=list()
        for j in range(len(dirlist)):
            try:
                if dirlist[j][-5:]=='.json':
                    #this assumes all .json in the folder are transcript (safe assumption if only .wav files)
                    jsonfile=dirlist[j]
                    features=json.load(open(jsonfile))['features']
                    one.append(features)
            except:
                pass 
            
        for j in range(len(dirlist)):
            if dirlist[j][-4:] in ['.mp3', '.wav'] and dirlist[j][0:-4]+'.json' not in dirlist and os.path.getsize(dirlist[j])>500:
                try:
                    #get wavefile
                    wavfile=dirlist[j]
                    print('%s - featurizing %s'%(name.upper(),wavfile))
                    #obtain features 
                    features=audio_features(wavfile)
                    print(features)
                    #append to list 
                    one.append(features.tolist())
                    #save intermediate .json just in case
                    data={
                        'features':features.tolist(),
                        }
                    jsonfile=open(dirlist[j][0:-4]+'.json','w')
                    json.dump(data,jsonfile)
                    jsonfile.close()
                except:
                    print('error')
            else:
                pass

        features_list.append(one)

    # randomly shuffle lists
    feature_list2=list()
    feature_lengths=list()
    for i in range(len(features_list)):
        one=features_list[i]
        random.shuffle(one)
        feature_list2.append(one)
        feature_lengths.append(len(one))

    # remember folderlist has all the labels
    
    min_num=np.amin(feature_lengths)
    #make sure they are the same length (For later) - this avoid errors
    while min_num*len(folderlist) != np.sum(feature_lengths):
        for i in range(len(folderlist)):
            while len(feature_list2[i])>min_num:
                print('%s is %s more than %s, balancing...'%(folderlist[i].upper(),str(len(feature_list2[i])-int(min_num)),'min value'))
                feature_list2[i].pop()
        feature_lengths=list()
        for i in range(len(feature_list2)):
            one=feature_list2[i]
            feature_lengths.append(len(one))
        
    #now write to json
    data={}
    for i in range(len(folderlist)):
        data.update({folderlist[i]:feature_list2[i]})
    
    os.chdir(dir3)
        
    jsonfile=open(jsonfilename,'w')
    json.dump(data,jsonfile)
    jsonfile.close()
    
else:
    pass

# DATA PREPROCESSING
#############################################################

# note that this assumes a classification problem based on total number of classes 
os.chdir(cur_dir)

#load data - can do this through loading .txt or .json files
#json file must have 'message' field 
data=json.loads(open(jsonfilename).read())

classes=list(data)
features=list()
labels=list()
for i in range(len(classes)):
    for j in range(len(data[classes[i]])):
        feature=data[classes[i]][j]
        features.append(feature)
        labels.append(classes[i])

train_set, test_set, train_labels, test_labels = train_test_split(features,
                                                                  labels,
                                                                  test_size=testing_set,
                                                                  random_state=42)

try:
    os.chdir(model_dir)
except:
    os.mkdir(model_dir)
    os.chdir(model_dir)
    
g=open(modelname+'_training_data.txt','w')
g.write('train labels'+'\n\n'+str(train_labels)+'\n\n')
g.write('test labels'+'\n\n'+str(test_labels)+'\n\n')
g.close()

training_data=open(modelname+'_training_data.txt').read()

# MODEL OPTIMIZATION / SAVE TO DISK 
#################################################################            
selectedfeature='audio features (mfcc coefficients).'
min_num=len(data[classes[0]])
[audio_model, audio_acc, audio_summary, data]=optimizemodel_sc(train_set,train_labels,test_set,test_labels,modelname,classes,testing_set,min_num,selectedfeature,training_data)

g=open(modelname+'.txt','w')
g.write(audio_summary)
g.close()

g2=open(modelname+'.json','w')
json.dump(data,g2)
g2.close()
              
print(audio_model)
print(audio_acc)
