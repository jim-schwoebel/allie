'''
Author: @Jim Schwoebel
Title: Train_mixeclassifykeras

This script takes in two folders filled with .wav files, folder A and folder B, and classifies them with nlx-textclassify features.

It upgrades train_textclassify2.py from simple classification techniques to deep learning approaches.

(C) 2018, NeuroLex Laboratories 

'''

import speech_recognition as sr
from pydub import AudioSegment
import librosa
import getpass 
import numpy as np
import random, os, json
import nltk 
from nltk import word_tokenize 
from nltk.classify import apply_features, SklearnClassifier, maxent
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
import keras.models
from keras import layers 
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout
from textblob import TextBlob
from operator import itemgetter
import getpass
import numpy as np
import pickle
import datetime 
import time 

#INITIALIZE VARIABLES 
#####################################################################
#size of each embedding vector
#(100 is used as default to reduce dimensionality)
size=100
#get start time to calculate execution time 
start=time.time()

#INITIALIZE FUNCTIONS
#####################################################################
#FEATURIZE .WAV FILES WITH AUDIO FEATURES --> MAKE JSON (if needed)
#############################################################
name1=input('folder name 1 \n')
name2=input('folder name 2 \n')
modelname=name1+'_'+name2+'_dl_text'
jsonfilename=name1+'_'+name2+'.json'
dir3='/Users/'+getpass.getuser()+'/nlx-model/nlx-textmodelkeras/'

try:
    os.chdir(dir3)
except:
    os.mkdir(dir3)
    os.chdir(dir3)

if jsonfilename not in os.listdir():

    try:
            
        g=json.load(jsonfile)

        #pass on this pre-loaded thing because the embedding is different
    
##        os.chdir('/Users/'+getpass.getuser()+'/nlx-model/nlx-textmodel')
##        g=json.load(jsonfile)
##        os.chdir('/Users/'+getpass.getuser()+'/nlx-model/nlx-textmodelkeras')
        
    except:
       #define some helper functions to featurize audio into proper form
        print('cannot load previous file')
        print('feauturizing')
        
        def featurize(wavfile):
            #initialize features 
            hop_length = 512
            n_fft=2048
            #load file 
            y, sr = librosa.load(wavfile)
            #extract mfcc coefficients 
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc) 
            #extract mean, standard deviation, min, and max value in mfcc frame, do this across all mfccs
            mfcc_features=np.array([np.mean(mfcc[0]),np.std(mfcc[0]),np.amin(mfcc[0]),np.amax(mfcc[0]),
                                    np.mean(mfcc[1]),np.std(mfcc[1]),np.amin(mfcc[1]),np.amax(mfcc[1]),
                                    np.mean(mfcc[2]),np.std(mfcc[2]),np.amin(mfcc[2]),np.amax(mfcc[2]),
                                    np.mean(mfcc[3]),np.std(mfcc[3]),np.amin(mfcc[3]),np.amax(mfcc[3]),
                                    np.mean(mfcc[4]),np.std(mfcc[4]),np.amin(mfcc[4]),np.amax(mfcc[4]),
                                    np.mean(mfcc[5]),np.std(mfcc[5]),np.amin(mfcc[5]),np.amax(mfcc[5]),
                                    np.mean(mfcc[6]),np.std(mfcc[6]),np.amin(mfcc[6]),np.amax(mfcc[6]),
                                    np.mean(mfcc[7]),np.std(mfcc[7]),np.amin(mfcc[7]),np.amax(mfcc[7]),
                                    np.mean(mfcc[8]),np.std(mfcc[8]),np.amin(mfcc[8]),np.amax(mfcc[8]),
                                    np.mean(mfcc[9]),np.std(mfcc[9]),np.amin(mfcc[9]),np.amax(mfcc[9]),
                                    np.mean(mfcc[10]),np.std(mfcc[10]),np.amin(mfcc[10]),np.amax(mfcc[10]),
                                    np.mean(mfcc[11]),np.std(mfcc[11]),np.amin(mfcc[11]),np.amax(mfcc[11]),
                                    np.mean(mfcc[12]),np.std(mfcc[12]),np.amin(mfcc[12]),np.amax(mfcc[12]),
                                    np.mean(mfcc_delta[0]),np.std(mfcc_delta[0]),np.amin(mfcc_delta[0]),np.amax(mfcc_delta[0]),
                                    np.mean(mfcc_delta[1]),np.std(mfcc_delta[1]),np.amin(mfcc_delta[1]),np.amax(mfcc_delta[1]),
                                    np.mean(mfcc_delta[2]),np.std(mfcc_delta[2]),np.amin(mfcc_delta[2]),np.amax(mfcc_delta[2]),
                                    np.mean(mfcc_delta[3]),np.std(mfcc_delta[3]),np.amin(mfcc_delta[3]),np.amax(mfcc_delta[3]),
                                    np.mean(mfcc_delta[4]),np.std(mfcc_delta[4]),np.amin(mfcc_delta[4]),np.amax(mfcc_delta[4]),
                                    np.mean(mfcc_delta[5]),np.std(mfcc_delta[5]),np.amin(mfcc_delta[5]),np.amax(mfcc_delta[5]),
                                    np.mean(mfcc_delta[6]),np.std(mfcc_delta[6]),np.amin(mfcc_delta[6]),np.amax(mfcc_delta[6]),
                                    np.mean(mfcc_delta[7]),np.std(mfcc_delta[7]),np.amin(mfcc_delta[7]),np.amax(mfcc_delta[7]),
                                    np.mean(mfcc_delta[8]),np.std(mfcc_delta[8]),np.amin(mfcc_delta[8]),np.amax(mfcc_delta[8]),
                                    np.mean(mfcc_delta[9]),np.std(mfcc_delta[9]),np.amin(mfcc_delta[9]),np.amax(mfcc_delta[9]),
                                    np.mean(mfcc_delta[10]),np.std(mfcc_delta[10]),np.amin(mfcc_delta[10]),np.amax(mfcc_delta[10]),
                                    np.mean(mfcc_delta[11]),np.std(mfcc_delta[11]),np.amin(mfcc_delta[11]),np.amax(mfcc_delta[11]),
                                    np.mean(mfcc_delta[12]),np.std(mfcc_delta[12]),np.amin(mfcc_delta[12]),np.amax(mfcc_delta[12])])
            
            return mfcc_features

        def exportfile(newAudio,time1,time2,filename,i):
            #Exports to a wav file in the current path.
            newAudio2 = newAudio[time1:time2]
            g=os.listdir()
            if filename[0:-4]+'_'+str(i)+'.wav' in g:
                filename2=str(i)+'_segment'+'.wav'
                print('making %s'%(filename2))
                newAudio2.export(filename2,format="wav")
            else:
                filename2=str(i)+'.wav'
                print('making %s'%(filename2))
                newAudio2.export(filename2, format="wav")

            return filename2 

        def audio_time_features(filename):
            #recommend >0.50 seconds for timesplit 
            timesplit=0.50
            hop_length = 512
            n_fft=2048
            
            y, sr = librosa.load(filename)
            duration=float(librosa.core.get_duration(y))
            
            #Now splice an audio signal into individual elements of 100 ms and extract
            #all these features per 100 ms
            segnum=round(duration/timesplit)
            deltat=duration/segnum
            timesegment=list()
            time=0

            for i in range(segnum):
                #milliseconds
                timesegment.append(time)
                time=time+deltat*1000

            newAudio = AudioSegment.from_wav(filename)
            filelist=list()
            
            for i in range(len(timesegment)-1):
                filename=exportfile(newAudio,timesegment[i],timesegment[i+1],filename,i)
                filelist.append(filename)

                featureslist=np.array([0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0,
                                       0,0,0,0])
            
            #save 100 ms segments in current folder (delete them after)
            for j in range(len(filelist)):
                try:
                    features=featurize(filelist[i])
                    featureslist=featureslist+features 
                    os.remove(filelist[j])
                except:
                    print('error splicing')
                    featureslist.append('silence')
                    os.remove(filelist[j])

            #now scale the featureslist array by the length to get mean in each category
            featureslist=featureslist/segnum
            
            return featureslist

        def textfeatures(transcript):
            #alphabetical features 
            a=transcript.count('a')
            b=transcript.count('b')
            c=transcript.count('c')
            d=transcript.count('d')
            e=transcript.count('e')
            f=transcript.count('f')
            g_=transcript.count('g')
            h=transcript.count('h')
            i=transcript.count('i')
            j=transcript.count('j')
            k=transcript.count('k')
            l=transcript.count('l')
            m=transcript.count('m')
            n=transcript.count('n')
            o=transcript.count('o')
            p=transcript.count('p')
            q=transcript.count('q')
            r=transcript.count('r')
            s=transcript.count('s')
            t=transcript.count('t')
            u=transcript.count('u')
            v=transcript.count('v')
            w=transcript.count('w')
            x=transcript.count('x')
            y=transcript.count('y')
            z=transcript.count('z')
            space=transcript.count(' ')

            #numerical features and capital letters 
            num1=transcript.count('0')+transcript.count('1')+transcript.count('2')+transcript.count('3')+transcript.count('4')+transcript.count('5')+transcript.count('6')+transcript.count('7')+transcript.count('8')+transcript.count('9')
            num2=transcript.count('zero')+transcript.count('one')+transcript.count('two')+transcript.count('three')+transcript.count('four')+transcript.count('five')+transcript.count('six')+transcript.count('seven')+transcript.count('eight')+transcript.count('nine')+transcript.count('ten')
            number=num1+num2
            capletter=sum(1 for c in transcript if c.isupper())

            #part of speech 
            text=word_tokenize(transcript)
            g=nltk.pos_tag(transcript)
            cc=0
            cd=0
            dt=0
            ex=0
            in_=0
            jj=0
            jjr=0
            jjs=0
            ls=0
            md=0
            nn=0
            nnp=0
            nns=0
            pdt=0
            pos=0
            prp=0
            prp2=0
            rb=0
            rbr=0
            rbs=0
            rp=0
            to=0
            uh=0
            vb=0
            vbd=0
            vbg=0
            vbn=0
            vbp=0
            vbp=0
            vbz=0
            wdt=0
            wp=0
            wrb=0
            
            for i in range(len(g)):
                if g[i][1] == 'CC':
                    cc=cc+1
                elif g[i][1] == 'CD':
                    cd=cd+1
                elif g[i][1] == 'DT':
                    dt=dt+1
                elif g[i][1] == 'EX':
                    ex=ex+1
                elif g[i][1] == 'IN':
                    in_=in_+1
                elif g[i][1] == 'JJ':
                    jj=jj+1
                elif g[i][1] == 'JJR':
                    jjr=jjr+1                   
                elif g[i][1] == 'JJS':
                    jjs=jjs+1
                elif g[i][1] == 'LS':
                    ls=ls+1
                elif g[i][1] == 'MD':
                    md=md+1
                elif g[i][1] == 'NN':
                    nn=nn+1
                elif g[i][1] == 'NNP':
                    nnp=nnp+1
                elif g[i][1] == 'NNS':
                    nns=nns+1
                elif g[i][1] == 'PDT':
                    pdt=pdt+1
                elif g[i][1] == 'POS':
                    pos=pos+1
                elif g[i][1] == 'PRP':
                    prp=prp+1
                elif g[i][1] == 'PRP$':
                    prp2=prp2+1
                elif g[i][1] == 'RB':
                    rb=rb+1
                elif g[i][1] == 'RBR':
                    rbr=rbr+1
                elif g[i][1] == 'RBS':
                    rbs=rbs+1
                elif g[i][1] == 'RP':
                    rp=rp+1
                elif g[i][1] == 'TO':
                    to=to+1
                elif g[i][1] == 'UH':
                    uh=uh+1
                elif g[i][1] == 'VB':
                    vb=vb+1
                elif g[i][1] == 'VBD':
                    vbd=vbd+1
                elif g[i][1] == 'VBG':
                    vbg=vbg+1
                elif g[i][1] == 'VBN':
                    vbn=vbn+1
                elif g[i][1] == 'VBP':
                    vbp=vbp+1
                elif g[i][1] == 'VBZ':
                    vbz=vbz+1
                elif g[i][1] == 'WDT':
                    wdt=wdt+1
                elif g[i][1] == 'WP':
                    wp=wp+1
                elif g[i][1] == 'WRB':
                    wrb=wrb+1

            #sentiment
            tblob=TextBlob(transcript)
            polarity=float(tblob.sentiment[0])
            subjectivity=float(tblob.sentiment[1])

            #word repeats
            words=transcript.split()
            newlist=transcript.split()
            repeat=0
            for i in range(len(words)):
                newlist.remove(words[i])
                if words[i] in newlist:
                    repeat=repeat+1 
            
            featureslist=np.array([a,b,c,d,
                                   e,f,g_,h,
                                   i,j,k,l,
                                   m,n,o,p,
                                   q,r,s,t,
                                   u,v,w,x,
                                   y,z,space,number,
                                   capletter,cc,cd,dt,
                                   ex,in_,jj,jjr,
                                   jjs,ls,md,nn,
                                   nnp,nns,pdt,pos,
                                   prp,prp2,rbr,rbs,
                                   rp,to,uh,vb,
                                   vbd,vbg,vbn,vbp,
                                   vbp,vbz,wdt,wp,
                                   wrb,polarity,subjectivity,repeat])
            
            return featureslist 
                                   
        def transcribe(wavfile):
            r = sr.Recognizer()
            # use wavfile as the audio source (must be .wav file)
            with sr.AudioFile(wavfile) as source:
                #extract audio data from the file
                audio = r.record(source)                    

            transcript=r.recognize_sphinx(audio)
            print(transcript)
            return transcript

        #104 features obtained by 
        #np.append(featurize(filename),audio_time_series(filename),textfeatures(filename))

        #load a folder, transcribe the data, make all the data into .txt files
        #load all the files into .json database (one,two)
        dir1='/Users/'+getpass.getuser()+'/nlx-model/nlx-textmodelkeras/'+name1

        g='error'
        while g == 'error':
            try:
                g='noterror'
                os.chdir(dir1)
            except:
                g='error'
                print('directory not recognized')
                dir1=input('input first directory path')

        dir2='/Users/'+getpass.getuser()+'/nlx-model/nlx-textmodelkeras/'+name2

        g='error'
        while g == 'error':
            try:
                g='noterror'
                os.chdir(dir2)
            except:
                g='error'
                print('directory not recognized')
                dir2=input('input second directory path')

        #now go through each directory and featurize the samples and save them as .json files
        os.chdir(dir1)
            
        dirlist1=os.listdir()

        #if broken session, load all previous transcripts
        #this reduces costs if tied to GCP
        one=list()
        for i in range(len(dirlist1)):
            try:
                if dirlist1[i][-5:]=='.json':
                    #this assumes all .json in the folder are transcript (safe assumption if only .wav files)
                    jsonfile=dirlist1[i]
                    features=json.load(open(jsonfile))['features']
                    one.append(features)
            except:
                pass 
                
        for i in range(len(dirlist1)):
            if dirlist1[i][-4:]=='.wav' and dirlist1[i][0:-4]+'.json'not in dirlist1 and os.path.getsize(dirlist1[i])>500:
                #loop through files and get features
                #try:
                wavfile=dirlist1[i]
                print('%s - featurizing %s'%(name1.upper(),wavfile))
                #ontain features 
                #try:
                features=textfeatures(transcribe(wavfile))
                print(features)
                #append to list 
                one.append(features.tolist())
                #save intermediate .json just in case
                data={
                    'features':features.tolist(),
                    }
                jsonfile=open(dirlist1[i][0:-4]+'.json','w')
                json.dump(data,jsonfile)
                jsonfile.close()
##                except:
##                    pass
            else:
                pass 

        #repeat same process in other directory
        os.chdir(dir2)
            
        dirlist2=os.listdir()

        two=list()
        
        for i in range(len(dirlist2)):
            try:
                if dirlist2[i][-5:]=='.json':
                    #this assumes all .json in the folder are transcript (safe assumption if only .wav files)
                    jsonfile=dirlist2[i]
                    features=json.load(open(jsonfile))['features']
                    two.append(features)                
            except:
                pass
                
        for i in range(len(dirlist2)):
            if dirlist2[i][-4:]=='.wav' and dirlist2[i][0:-4]+'.json' not in dirlist2 and os.path.getsize(dirlist2[i])>500:
                #loop through files and get features 
                try:
                    wavfile=dirlist2[i]
                    print('%s - featurizing %s'%(name2.upper(),wavfile))
                    #obtain features 
                    features=textfeatures(transcribe(wavfile))
                    print(features)
                    #append to list 
                    two.append(features.tolist())
                    #save intermediate .json just in case
                    data={
                        'features':features.tolist(),
                        }
                    jsonfile=open(dirlist2[i][0:-4]+'.json','w')
                    json.dump(data,jsonfile)
                    jsonfile.close()
                except:
                    pass
            else:
                pass

        #randomly shuffle one and two
        random.shuffle(one)
        random.shuffle(two)
        
        #make sure they are the same length (For later) - this avoid errors
        while len(one)>len(two):
            print('%s is %s more than %s, balancing...'%(name1.upper(),str(len(one)-len(two)),name2.upper()))
            one.pop()
        while len(two)>len(one):
            print('%s is %s more than %s, balancing...'%(name2.upper(),str(len(two)-len(one)),name1.upper()))
            two.pop()
            
        #now write to json
        data={
            name1:one,
            name2:two,
            }

        os.chdir(dir3)
        jsonfile=open(name1+'_'+name2+'.json','w')
        json.dump(data,jsonfile)
        jsonfile.close()


#LOAD AND BALANCE DATASETS
#####################################################################
#get output vector of 1 (similarity)
#load this using some form of .json
#note this assumes a binary classification problem A/B
try:
    g=json.load(open(name1+'_'+name2+'.json'))
except:
    print('error loading .json')
    
#don't try this because output embeddings are different 
##    try:
##        
##        os.chdir('/Users/'+getpass.getuser()+'/nlx-model/nlx-textmodel')
##        g=json.load(open(name1+'_'+name2+'.json'))
##        os.chdir('/Users/'+getpass.getuser()+'/nlx-model/nlx-textmodelkeras')
##        print('loading transcripts from nlx-textmodel directory')
                      
        
s1_temp =g[name1]
s2_temp= g[name2]

s1list=list()
s2list=list()

#make into well-formatted numpy arrays 
for i in range(len(s1_temp)):
    s1list.append(np.array(g[name1][i]))
                  
for i in range(len(s2_temp)):
    s2list.append(np.array(g[name2][i]))

s1=s1list
s2=s2list

#TEST AND TRAIN SET GENERATION 
#################################################################
#Now generate train and test sets (1/2 left out for testing)
#randomize message labels
labels=list()

for i in range(len(s1)):
    labels.append([s1[i],0])
    
for i in range(len(s2)):
    labels.append([s2[i],1])

random.shuffle(labels)
half=int(len(labels)/2)

#make sure we featurize all these train_set2s 
train_set2=list()
labels_train_set2=list()
test_set2=list()
labels_test_set2=list()

for i in range(half):
    embedding=labels[half][0]
    train_set2.append(embedding)
    labels_train_set2.append(labels[half][1])

for i in range(half):
    #repeat same process for embeddings
    embedding=labels[half+i][0]
    test_set2.append(embedding)
    labels_test_set2.append(labels[half+i][1])

#MODEL OPTIMIZATION 
#################################################################
#make classifier/test classifier (these are all easy methods)

# DATA PRE-PROCESSING 
############################################################################
x_train = np.array(train_set2)
y_train = np.array(labels_train_set2)
x_test = np.array(test_set2)
y_test = np.array(labels_test_set2)

# MAKE MODEL
############################################################################
model = Sequential()
model.add(Dense(64, input_dim=len(x_train[0]), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=100,
          batch_size=128)

# EVALUATE MODEL / PREDICT OUTPUT 
############################################################################
score = model.evaluate(x_test, y_test, batch_size=128)

print("\n final %s: %.2f%% \n" % (model.metrics_names[1], score[1]*100))
print(model.predict(x_train[0][np.newaxis,:]))

model = Sequential()
model.add(Dense(64, input_dim=len(x_train[0]), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train+x_test, y_train+y_test,
          epochs=100,
          batch_size=128)

#SAVE TO DISK
############################################################################
try:
    os.chdir(os.getcwd()+'/models')
except:
    os.mkdir(os.getcwd()+'/models')
    os.chdir(os.getcwd()+'/models')

# serialize model to JSON
model_json = model.to_json()
with open(modelname+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(modelname+".h5")
print("\n Saved %s.json model to disk"%(modelname))

# SUMMARIZE RESULTS
############################################################################
execution=time.time()-start
print('summarizing data...')
g=open(modelname+'.txt','w')
g.write('SUMMARY OF MODEL')
g.write('\n\n')
g.write('Keras-based implementation of a neural network, 64 input text features, 1 output feature; binary classification, 2 layers (relu | sigmoid activation functions), loss=binary_crossentropy, optimizer=rmsprop')
g.write('\n\n')
g.write('MODEL FILE NAME: \n\n %s.json | %s.h5'%(modelname,modelname))
g.write('\n\n')
g.write('DATE CREATED: \n\n %s'%(datetime.datetime.now()))
g.write('\n\n')
g.write('EXECUTION TIME: \n\n %s\n\n'%(str(execution)))
g.write('GROUPS: \n\n')
g.write('Group 1: %s (%s training, %s testing)'%(name1,str(int(len(train_set2)/2)),str(int(len(test_set2)/2))))
g.write('\n')
g.write('Group 2: %s (%s training, %s testing)'%(name2,str(int(len(labels_train_set2)/2)),str(int(len(labels_test_set2)/2))))
g.write('\n\n')
g.write('FEATURES: \n\n Word2Vec representation of %s and %s (200 features)'%(name1,name2))
g.write('\n\n')
g.write('MODEL ACCURACY: \n\n')
g.write('%s: %s \n\n'%(str('accuracy'),str(score[1]*100)))
g.write('(C) 2018, NeuroLex Laboratories')
g.close()


##APPLY CLASSIFIER TO PREDICT DATA
############################################################################
#This is an automated unit test to save and load the model to disk
#ensures you can call the model later 
print('testing loaded model\n')
# load json and create model
json_file = open(modelname+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(modelname+".h5")
print("Loaded model from disk \n")
print(loaded_model.predict(x_train[0][np.newaxis,:]))

### evaluate loaded model on test data
##loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
##score = loaded_model.evaluate(X, Y, verbose=0)

#OUTPUT CLASS
#print(model.predict_classes(x_train[0][np.newaxis,:]))
