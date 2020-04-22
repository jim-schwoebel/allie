'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: audio_time_features 

#Note that another library exists for time-series features
(e.g. at 100 ms timescale). Uses audio_features library.

See the documentation here: https://librosa.github.io/librosa/

All these features are represented as the mean, standard deviation, variance,
median, min, and max.
'''
import librosa
import numpy as np
from pydub import AudioSegment
import os
from . import audio_features

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
      
def audio_time_features(filename,timesplit):
  #recommend >0.50 seconds for timesplit (timesplit > 0.50)
  
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

  featureslist=list()
  
  #save 100 ms segments in current folder (delete them after)
  for j in range(len(filelist)):
    try:
      features=audio_features.audio_features(filelist[i])
      featureslist.append(features)
      os.remove(filelist[j])
    except:
      print('error splicing')
      featureslist.append('silence')
      os.remove(filelist[j])
      
  #outputfeatures 

  return [duration, segnum, featureslist]

#test - recommended settings @ 0.50 seconds 
##os.chdir('/Users/jimschwoebel/Desktop/audiotest')
##output=audio_time_features('test.wav',0.500)
#timessplit=secs 
  
