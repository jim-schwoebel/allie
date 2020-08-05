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
                           

This will featurize folders of audio files if the default_audio_features = ['standard_features']

A standard feature array extracted using LibROSA's library. 
'''
import librosa, os, uuid
import numpy as np 
from pydub import AudioSegment 

def audio_featurize(wavfile):
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
        filename2=str(uuid.uuid4())+'_segment'+'.wav'
        print('making %s'%(filename2))
        newAudio2.export(filename2,format="wav")
    else:
        filename2=str(uuid.uuid4())+'.wav'
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

    if filename[-4:]=='.wav':
        newAudio = AudioSegment.from_wav(filename)
    elif filename[-4:]=='.mp3':
        newAudio = AudioSegment.from_mp3(filename)
        
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
            features=audio_featurize(filelist[i])
            featureslist=featureslist+features 
            os.remove(filelist[j])
        except:
            print('error splicing')
            featureslist.append('silence')
            os.remove(filelist[j])

    # now scale the featureslist array by the length to get mean in each category
    featureslist=featureslist/segnum

    return featureslist

def standard_featurize(filename):
    features=np.append(audio_featurize(filename), audio_time_features(filename))
    # labels
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


    return features, labels 
