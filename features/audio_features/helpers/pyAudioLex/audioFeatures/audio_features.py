'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: audio_features 

#These are all audio features extracted with librosa library.

See the documentation here: https://librosa.github.io/librosa/

All these features are represented as the mean, standard deviation, variance,
median, min, and max.

Note that another library exists for time-series features (e.g. at 100 ms timescale).
'''

import librosa
import numpy as np
import os

def statlist(veclist):
  
  newlist=list()
  
  #fingerprint statistical features
  #append each with mean, std, var, median, min, and max
  if len(veclist)>100:
    newlist=[float(np.mean(veclist)),float(np.std(veclist)),float(np.var(veclist)),
         float(np.median(veclist)),float(np.amin(veclist)),float(np.amax(veclist))]
      
  else:
    for i in range(len(veclist)):
      newlist.append([float(np.mean(veclist[i])),float(np.std(veclist[i])),float(np.var(veclist[i])),
              float(np.median(veclist[i])),float(np.amin(veclist[i])),float(np.amax(veclist[i]))])
      
  return newlist

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
  mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
  mfcc_delta = librosa.feature.delta(mfcc)
  beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)
  chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
  beat_chroma = librosa.util.sync(chromagram,
                beat_frames,
                aggregate=np.median)
  beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
  zero_crossings = librosa.zero_crossings(y)
  zero_crossing_time = librosa.feature.zero_crossing_rate(y)
  spectral_centroid = librosa.feature.spectral_centroid(y)
  spectral_bandwidth = librosa.feature.spectral_bandwidth(y)
  spectral_contrast = librosa.feature.spectral_contrast(y)
  spectral_rolloff = librosa.feature.spectral_rolloff(y)
  rmse=librosa.feature.rmse(y)
  poly_features=librosa.feature.poly_features(y)
  chroma_stft = librosa.feature.chroma_stft(y)
  chroma_cens = librosa.feature.chroma_cens(y)
  tonnetz=librosa.feature.tonnetz(y)
  
  mfcc_all=statlist(mfcc)
  mfccd_all=statlist(mfcc_delta)
  bmfccd_all=statlist(beat_mfcc_delta)
  cg_all=statlist(chromagram)
  bc_all=statlist(beat_chroma)
  bf_all=statlist(beat_features)
  zc_all=statlist(zero_crossings)
  sc_all=statlist(spectral_centroid)
  sb_all=statlist(spectral_bandwidth)
  sc_all=statlist(spectral_contrast)
  sr_all=statlist(spectral_rolloff)
  rmse_all=statlist(rmse)
  pf_all=statlist(poly_features)
  cstft_all=statlist(chroma_stft)
  ccens_all=statlist(chroma_cens)
  tonnetz_all=statlist(tonnetz)
  
  return [duration,float(tempo),beat_frames.tolist(),beat_times.tolist(),mfcc_all,
      mfccd_all,bmfccd_all,cg_all,bc_all,bf_all,zc_all,sc_all,sb_all,
      sc_all,sr_all,rmse_all,pf_all,cstft_all,ccens_all,tonnetz_all]

#TEST 
#import os 
#os.chdir('/Users/jimschwoebel/Desktop')
#output=audio_features('test.wav')
