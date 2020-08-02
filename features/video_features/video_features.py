import numpy as np
import cv2, os, random, json, sys, getpass, pickle, datetime, time, librosa, shutil, gensim, nltk
from nltk import word_tokenize 
from nltk.classify import apply_features, SklearnClassifier, maxent
import speech_recognition as sr
from pydub import AudioSegment
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from textblob import TextBlob
from operator import itemgetter
from matplotlib import pyplot as plt
from PIL import Image
import skvideo.io
import skvideo.motion
import skvideo.measure
from moviepy.editor import VideoFileClip
from matplotlib import pyplot as plt
from pydub import AudioSegment

def prev_dir(directory):
    g=directory.split('/')
    # print(g)
    lastdir=g[len(g)-1]
    i1=directory.find(lastdir)
    directory=directory[0:i1]
    return directory

#### to extract tesseract features 
curdir=os.getcwd()
import helpers.tesseract_features as tf 
os.chdir(curdir)

# DEFINE HELPER FUNCTIONS
#############################################################
def convert(file):
    clip = VideoFileClip(file)
    duration = clip.duration

    if duration < 30:
        if file[-4:] in ['.mov','.avi','.flv','.wmv']:
            filename=file[0:-4]+'.mp4'
            os.system('ffmpeg -i %s -an %s'%(file,filename))
            os.remove(file)
        elif file[-4:] == '.mp4':
            filename=file
        else:
            filename=file 
            os.remove(file)
    else:
        filename=file
        os.remove(file)

    return filename

def haar_featurize(cur_dir, haar_dir, img):

    os.chdir(haar_dir)
    # load image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # assumes all files of haarcascades are in current directory 

    one = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    one = one.detectMultiScale(gray, 1.3, 5)
    one = len(one)
              
    two = cv2.CascadeClassifier('haarcascade_eye.xml')
    two = two.detectMultiScale(gray, 1.3, 5)
    two = len(two)
              
    three = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
    three = three.detectMultiScale(gray, 1.3, 5)
    three = len(three)
              
    four = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
    four = four.detectMultiScale(gray, 1.3, 5)
    four = len(four)
              
    five = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
    five = five.detectMultiScale(gray, 1.3, 5)
    five = len(five)
              
    six = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    six = six.detectMultiScale(gray, 1.3, 5)
    six = len(six)
              
    seven = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    seven = seven.detectMultiScale(gray, 1.3, 5)
    seven = len(seven)
              
    eight = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eight = eight.detectMultiScale(gray, 1.3, 5)
    eight = len(eight)
              
    nine = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    nine = nine.detectMultiScale(gray, 1.3, 5)
    nine = len(nine)
              
    ten = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    ten = ten.detectMultiScale(gray, 1.3, 5)
    ten = len(ten)
              
    eleven = cv2.CascadeClassifier('haarcascade_licence_plate_rus_16stages.xml')
    eleven = eleven.detectMultiScale(gray, 1.3, 5)
    eleven = len(eleven)
              
    twelve = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
    twelve = twelve.detectMultiScale(gray, 1.3, 5)
    twelve = len(twelve)
              
    thirteen = cv2.CascadeClassifier('haarcascade_profileface.xml')
    thirteen = thirteen.detectMultiScale(gray, 1.3, 5)
    thirteen = len(thirteen)
              
    fourteen = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    fourteen = fourteen.detectMultiScale(gray, 1.3, 5)
    fourteen = len(fourteen)
              
    fifteen = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
    fifteen = fifteen.detectMultiScale(gray, 1.3, 5)
    fifteen = len(fifteen)
              
    sixteen = cv2.CascadeClassifier('haarcascade_smile.xml')
    sixteen = sixteen.detectMultiScale(gray, 1.3, 5)
    sixteen = len(sixteen)
              
    seventeen = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    seventeen = seventeen.detectMultiScale(gray, 1.3, 5)
    seventeen = len(seventeen)

    features=np.array([one,two,three,four,
                      five,six,seven,eight,
                      nine,ten,eleven,twelve,
                      thirteen,fourteen,fifteen,sixteen,
                      seventeen])
              
    labels=['haarcascade_eye_tree_eyeglasses','haarcascade_eye','haarcascade_frontalcatface_extended','haarcascade_frontalcatface',
            'haarcascade_frontalface_alt_tree','haarcascade_frontalface_alt','haarcascade_frontalface_alt2','haarcascade_frontalface_default',
            'haarcascade_fullbody','haarcascade_lefteye_2splits','haarcascade_licence_plate_rus_16stages','haarcascade_lowerbody',
            'haarcascade_profileface','haarcascade_righteye_2splits','haarcascade_russian_plate_number','haarcascade_smile',
            'haarcascade_upperbody']

    os.chdir(cur_dir)
              
    return features, labels 
                  
def image_featurize(cur_dir,haar_dir,file):

    # initialize label array 
    labels=list()
    # only featurize files that are .jpeg, .jpg, or .png (convert all to ping
    if file[-5:]=='.jpeg':
        filename=convert(file)
    elif file[-4:]=='.jpg':
        filename=convert(file)
    elif file[-4:]=='.png':
        filename=file 
    else:
        filename=file
              
    #only featurize .png files after conversion 
    if filename[-4:]=='.png':
        # READ IMAGE
        ########################################################
        img = cv2.imread(filename,1)

        # CALCULATE BASIC FEATURES (rows, columns, pixels)
        ########################################################
        #rows, columns, pixel number
        rows=img.shape[1]
        columns=img.shape[2]
        pixels=img.size

        basic_features=np.array([rows,columns,pixels])
        labels=labels+['rows', 'columns', 'pixels']
        # HISTOGRAM FEATURES (avg, stdev, min, max)
        ########################################################
        #blue
        blue_hist=cv2.calcHist([img],[0],None,[256],[0,256])
        blue_mean=np.mean(blue_hist)
        blue_std=np.std(blue_hist)
        blue_min=np.amin(blue_hist)
        blue_max=np.amax(blue_hist)
        #green
        green_hist=cv2.calcHist([img],[1],None,[256],[0,256])
        green_mean=np.mean(green_hist)
        green_std=np.std(green_hist)
        green_min=np.amin(green_hist)
        green_max=np.amax(green_hist)
        #red
        red_hist=cv2.calcHist([img],[2],None,[256],[0,256])
        red_mean=np.mean(red_hist)
        red_std=np.std(red_hist)
        red_min=np.amin(red_hist)
        red_max=np.amax(red_hist)

        hist_features=[blue_mean,blue_std,blue_min,blue_max,
                       green_mean,green_std,green_min,green_max,
                       red_mean,red_std,red_min,red_max]
        hist_labels=['blue_mean','blue_std','blue_min','blue_max',
                     'green_mean','green_std','green_min','green_max',
                     'red_mean','red_std','red_min','red_max']

        hist_features=np.array(hist_features)

        features=np.append(basic_features,hist_features)
        labels=labels+hist_labels 

        # CALCULATE HAAR FEATURES
        ########################################################
        haar_features, haar_labels=haar_featurize(cur_dir,haar_dir,img)
        
        features=np.append(features,haar_features)
        labels=labels+haar_labels
        
        # EDGE FEATURES
        ########################################################
        # SIFT algorithm (scale invariant) - 128 features
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        (kps, des) = sift.detectAndCompute(gray, None)
        edges=des
        edge_features=np.zeros(len(edges[0]))
              
        for i in range(len(edges)):
            edge_features=edge_features+edges[i]

        edge_features=edge_features/(len(edges))
        edge_features=np.array(edge_features)
        edge_labels=list()
        for i in range(len(edge_features)):
            edge_labels.append('edge_feature_%s'%(str(i+1)))
        features=np.append(features,edge_features)
        labels=labels+edge_labels 
              
    else:
        os.remove(file)

    return features, labels

def featurize_audio(wavfile):
    #initialize features 
    hop_length = 512
    n_fft=2048
    #load file 
    y, sr = librosa.load(wavfile)
    #extract mfcc coefficients 
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc) 
    #extract mean, standard deviation, min, and max value in mfcc frame, do this across all mfccs
    features=np.array([np.mean(mfcc[0]),np.std(mfcc[0]),np.amin(mfcc[0]),np.amax(mfcc[0]),
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

    return features

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
            features=featurize_audio(filelist[i])
            featureslist=featureslist+features 
            os.remove(filelist[j])
        except:
            print('error splicing')
            os.remove(filelist[j])

    #now scale the featureslist array by the length to get mean in each category
    features=featureslist/segnum

    return features 

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

    features=np.array([a,b,c,d,
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
                       vbz,wdt,wp, wrb,polarity,subjectivity,repeat])

    labels=['a','b','c','d',
            'e','f','g_','h',
            'i','j','k','l',
            'm','n','o','p',
            'q','r','s','t',
            'u','v','w','x',
            'y','z','space','number',
            'capletter','cc','cd','dt',
            'ex','in_','jj','jjr',
            'jjs','ls','md','nn',
            'nnp','nns','pdt','pos',
            'prp','prp2','rbr','rbs',
            'rp','to','uh','vb',
            'vbd','vbg','vbn','vbp',
            'vbz','wdt','wp', 'polarity','subjectivity','repeat']
    
    return features, labels
                           
def transcribe(wavfile):
    try:
        r = sr.Recognizer()
        # use wavfile as the audio source (must be .wav file)
        with sr.AudioFile(wavfile) as source:
            #extract audio data from the file
            audio = r.record(source)                    

        transcript=r.recognize_sphinx(audio)
        print(transcript)
    except:
        transcript=''
    return transcript


# featurize only a random 20 second slice of the video (or 20 sec splices of videos)
def video_featurize(videofile, cur_dir,haar_dir):
    now=os.getcwd()
    # PREPROCESSING
    #############################################
    # metadata (should be .mp4)
    clip = VideoFileClip(videofile)
    duration = clip.duration
    videodata=skvideo.io.vread(videofile)
    frames, rows, cols, channels = videodata.shape
    metadata=skvideo.io.ffprobe(videofile)
    frame=videodata[0]
    r,c,ch=frame.shape

    try:
        os.mkdir('output')
        os.chdir('output')
        outputdir=os.getcwd()
    except:
        shutil.rmtree('output')
        os.mkdir('output')
        os.chdir('output')
        outputdir=os.getcwd()

    #write all the images every 10 frames in the video 
    for i in range(0,len(videodata),25):
        #row, col, channels
        skvideo.io.vwrite("output"+str(i)+".png", videodata[i])
        
    listdir=os.listdir()
    (r,c,ch)=cv2.imread(listdir[0]).shape
    img=np.zeros((r,c,ch))
    iterations=0
    #take first image as a background image 
    background=cv2.imread(listdir[1])
    image_features=np.zeros(160)
    image_features2=np.zeros(63)

    image_transcript=''
    for i in range(len(listdir)):
        if listdir[i][-4:]=='.png':
            os.chdir(outputdir)
            frame_new=cv2.imread(listdir[i])
            print(os.getcwd())
            print(listdir[i])
            print(frame)
            img=img+frame_new
            iterations=iterations+1
            image_features_temp, image_labels = image_featurize(cur_dir,haar_dir,listdir[i])
            os.chdir(outputdir)
            ttranscript, tfeatures, tlabels = tf.tesseract_featurize(listdir[i])
            image_transcript=image_transcript+ttranscript 
            image_features2=image_features2+tfeatures 
            image_features=image_features+image_features_temp
            #os.remove(listdir[i])

    # averaged image features
    image_features=(1/iterations)*image_features
    image_features2=(1/iterations)*image_features2

    # averaged image over background             
    img=(1/iterations)*img-background
    skvideo.io.vwrite("output.png", img)
    avg_image_features, image_labels =image_featurize(cur_dir,haar_dir, "output.png")
    # remove temp directory 
    os.chdir(now)
    
    # make wavfile from video file
    wavfile = videofile[0:-4]+'.wav'
    os.system('ffmpeg -i %s %s'%(videofile,wavfile))

    print('made wavfile in %s'%(str(os.getcwd())))

    # FEATURIZATION
    #############################################
    # audio features and time features 
    labels=list()
    audio_features=np.append(featurize_audio(wavfile),audio_time_features(wavfile))
    labels=['mfcc_1_mean_20ms','mfcc_1_std_20ms', 'mfcc_1_min_20ms', 'mfcc_1_max_20ms',
            'mfcc_2_mean_20ms','mfcc_2_std_20ms', 'mfcc_2_min_20ms', 'mfcc_2_max_20ms',
            'mfcc_3_mean_20ms','mfcc_3_std_20ms', 'mfcc_3_min_20ms', 'mfcc_3_max_20ms',
            'mfcc_4_mean_20ms','mfcc_4_std_20ms', 'mfcc_4_min_20ms', 'mfcc_4_max_20ms',
            'mfcc_5_mean_20ms','mfcc_5_std_20ms', 'mfcc_5_min_20ms', 'mfcc_5_max_20ms',
            'mfcc_6_mean_20ms','mfcc_6_std_20ms', 'mfcc_6_min_20ms', 'mfcc_6_max_20ms',
            'mfcc_7_mean_20ms','mfcc_7_std_20ms', 'mfcc_7_min_20ms', 'mfcc_7_max_20ms',
            'mfcc_8_mean_20ms','mfcc_8_std_20ms', 'mfcc_8_min_20ms', 'mfcc_8_max_20ms',
            'mfcc_9_mean_20ms','mfcc_9_std_20ms', 'mfcc_9_min_20ms', 'mfcc_9_max_20ms',
            'mfcc_10_mean_20ms','mfcc_10_std_20ms', 'mfcc_10_min_20ms', 'mfcc_10_max_20ms',
            'mfcc_11_mean_20ms','mfcc_11_std_20ms', 'mfcc_11_min_20ms', 'mfcc_11_max_20ms',
            'mfcc_12_mean_20ms','mfcc_12_std_20ms', 'mfcc_12_min_20ms', 'mfcc_12_max_20ms',
            'mfcc_13_mean_20ms','mfcc_13_std_20ms', 'mfcc_13_min_20ms', 'mfcc_13_max_20ms',
            'mfcc_1_delta_mean_20ms','mfcc_1_delta_std_20ms', 'mfcc_1_delta_min_20ms', 'mfcc_1_delta_max_20ms',
            'mfcc_2_delta_mean_20ms','mfcc_2_delta_std_20ms', 'mfcc_2_delta_min_20ms', 'mfcc_2_delta_max_20ms',
            'mfcc_3_delta_mean_20ms','mfcc_3_delta_std_20ms', 'mfcc_3_delta_min_20ms', 'mfcc_3_delta_max_20ms',
            'mfcc_4_delta_mean_20ms','mfcc_4_delta_std_20ms', 'mfcc_4_delta_min_20ms', 'mfcc_4_delta_max_20ms',
            'mfcc_5_delta_mean_20ms','mfcc_5_delta_std_20ms', 'mfcc_5_delta_min_20ms', 'mfcc_5_delta_max_20ms',
            'mfcc_6_delta_mean_20ms','mfcc_6_delta_std_20ms', 'mfcc_6_delta_min_20ms', 'mfcc_6_delta_max_20ms',
            'mfcc_7_delta_mean_20ms','mfcc_7_delta_std_20ms', 'mfcc_7_delta_min_20ms', 'mfcc_7_delta_max_20ms',
            'mfcc_8_delta_mean_20ms','mfcc_8_delta_std_20ms', 'mfcc_8_delta_min_20ms', 'mfcc_8_delta_max_20ms',
            'mfcc_9_delta_mean_20ms','mfcc_9_delta_std_20ms', 'mfcc_9_delta_min_20ms', 'mfcc_9_delta_max_20ms',
            'mfcc_10_delta_mean_20ms','mfcc_10_delta_std_20ms', 'mfcc_10_delta_min_20ms', 'mfcc_10_delta_max_20ms',
            'mfcc_11_delta_mean_20ms','mfcc_11_delta_std_20ms', 'mfcc_11_delta_min_20ms', 'mfcc_11_delta_max_20ms',
            'mfcc_12_delta_mean_20ms','mfcc_12_delta_std_20ms', 'mfcc_12_delta_min_20ms', 'mfcc_12_delta_max_20ms',
            'mfcc_13_delta_mean_20ms','mfcc_13_delta_std_20ms', 'mfcc_13_delta_min_20ms', 'mfcc_13_delta_max_20ms',
            'mfcc_1_mean_500ms','mfcc_1_std_500ms', 'mfcc_1_min_500ms', 'mfcc_1_max_500ms',
            'mfcc_2_mean_500ms','mfcc_2_std_500ms', 'mfcc_2_min_500ms', 'mfcc_2_max_500ms',
            'mfcc_3_mean_500ms','mfcc_3_std_500ms', 'mfcc_3_min_500ms', 'mfcc_3_max_500ms',
            'mfcc_4_mean_500ms','mfcc_4_std_500ms', 'mfcc_4_min_500ms', 'mfcc_4_max_500ms',
            'mfcc_5_mean_500ms','mfcc_5_std_500ms', 'mfcc_5_min_500ms', 'mfcc_5_max_500ms',
            'mfcc_6_mean_500ms','mfcc_6_std_500ms', 'mfcc_6_min_500ms', 'mfcc_6_max_500ms',
            'mfcc_7_mean_500ms','mfcc_7_std_500ms', 'mfcc_7_min_500ms', 'mfcc_7_max_500ms',
            'mfcc_8_mean_500ms','mfcc_8_std_500ms', 'mfcc_8_min_500ms', 'mfcc_8_max_500ms',
            'mfcc_9_mean_500ms','mfcc_9_std_500ms', 'mfcc_9_min_500ms', 'mfcc_9_max_500ms',
            'mfcc_10_mean_500ms','mfcc_10_std_500ms', 'mfcc_10_min_500ms', 'mfcc_10_max_500ms',
            'mfcc_11_mean_500ms','mfcc_11_std_500ms', 'mfcc_11_min_500ms', 'mfcc_11_max_500ms',
            'mfcc_12_mean_500ms','mfcc_12_std_500ms', 'mfcc_12_min_500ms', 'mfcc_12_max_500ms',
            'mfcc_13_mean_500ms','mfcc_13_std_500ms', 'mfcc_13_min_500ms', 'mfcc_13_max_500ms',
            'mfcc_1_delta_mean_500ms','mfcc_1_delta_std_500ms', 'mfcc_1_delta_min_500ms', 'mfcc_1_delta_max_500ms',
            'mfcc_2_delta_mean_500ms','mfcc_2_delta_std_500ms', 'mfcc_2_delta_min_500ms', 'mfcc_2_delta_max_500ms',
            'mfcc_3_delta_mean_500ms','mfcc_3_delta_std_500ms', 'mfcc_3_delta_min_500ms', 'mfcc_3_delta_max_500ms',
            'mfcc_4_delta_mean_500ms','mfcc_4_delta_std_500ms', 'mfcc_4_delta_min_500ms', 'mfcc_4_delta_max_500ms',
            'mfcc_5_delta_mean_500ms','mfcc_5_delta_std_500ms', 'mfcc_5_delta_min_500ms', 'mfcc_5_delta_max_500ms',
            'mfcc_6_delta_mean_500ms','mfcc_6_delta_std_500ms', 'mfcc_6_delta_min_500ms', 'mfcc_6_delta_max_500ms',
            'mfcc_7_delta_mean_500ms','mfcc_7_delta_std_500ms', 'mfcc_7_delta_min_500ms', 'mfcc_7_delta_max_500ms',
            'mfcc_8_delta_mean_500ms','mfcc_8_delta_std_500ms', 'mfcc_8_delta_min_500ms', 'mfcc_8_delta_max_500ms',
            'mfcc_9_delta_mean_500ms','mfcc_9_delta_std_500ms', 'mfcc_9_delta_min_500ms', 'mfcc_9_delta_max_500ms',
            'mfcc_10_delta_mean_500ms','mfcc_10_delta_std_500ms', 'mfcc_10_delta_min_500ms', 'mfcc_10_delta_max_500ms',
            'mfcc_11_delta_mean_500ms','mfcc_11_delta_std_500ms', 'mfcc_11_delta_min_500ms', 'mfcc_11_delta_max_500ms',
            'mfcc_12_delta_mean_500ms','mfcc_12_delta_std_500ms', 'mfcc_12_delta_min_500ms', 'mfcc_12_delta_max_500ms',
            'mfcc_13_delta_mean_500ms','mfcc_13_delta_std_500ms', 'mfcc_13_delta_min_500ms', 'mfcc_13_delta_max_500ms']
    # text features
    transcript = transcribe(wavfile)
    text_features, text_labels =textfeatures(transcript)
    labels=labels+text_labels 
    # video features
    video_features=np.append(image_features, avg_image_features)
    avg_image_labels=list()
    for i in range(len(image_labels)):
        avg_image_labels.append('avg_'+image_labels[i])

    avg_image_labels2=list()
    for i in range(len(tlabels)):
        avg_image_labels2.append('avg_imgtranscript_'+tlabels[i])

    video_labels=image_labels + avg_image_labels + avg_image_labels2
    labels=labels+video_labels 
    # other features
    other_features = [frames,duration]
    other_labels = ['frames', 'duration']
    labels=labels+other_labels 

    # append all the features together
    features = np.append(audio_features, text_features)
    features = np.append(features, image_features)
    features = np.append(features, image_features2)
    features = np.append(features, video_features)
    features = np.append(features, other_features)

    # remove all temp files
    try:
        os.remove(wavfile)
    except:
        pass
    try:
        shutil.rmtree('output')
    except:
        pass
    os.chdir(cur_dir)

    return features, labels, transcript, image_transcript 