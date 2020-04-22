'''
@package: pyAudioLex
@author: Drew Morris
@module: audio

Used to process all audio features based heavily on pyAudioAnalysis
'''

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import wave
import contextlib

# get duration
def get_duration(filepath):
  try:
    wavefile = wave.open(filepath, 'r')

    # see how long the file is
    with contextlib.closing(wavefile) as f:
      frames = f.getnframes()
      rate = f.getframerate()
      duration = frames / float(rate)

    return duration
  except:
    return 0.0

# process audio
def audio_featurize(wav):
  [Fs, x] = audioBasicIO.readAudioFile(wav)
  x = audioBasicIO.stereo2mono(x) # convert to mono
  F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)[0]
  print(F)
  results = {}
  results['ZCR'] = F[0].tolist()
  results['energy'] = F[1].tolist()
  results['entropy'] = F[2].tolist()
  results['spectral_centroid'] = F[3].tolist()
  results['spectral_spread'] = F[4].tolist()
  results['spectral_entropy'] = F[5].tolist()
  results['spectral_flux'] = F[6].tolist()
  results['spectral_rolloff'] = F[7].tolist()
  results['MFCC_1']=F[8].tolist()
  results['MFCC_2']=F[9].tolist()
  results['MFCC_3']=F[10].tolist()
  results['MFCC_4']=F[11].tolist()
  results['MFCC_5']=F[12].tolist()
  results['MFCC_6']=F[13].tolist()
  results['MFCC_7']=F[14].tolist()
  results['MFCC_8']=F[15].tolist()
  results['MFCC_9']=F[16].tolist()
  results['MFCC_10']=F[17].tolist()
  results['MFCC_11']=F[18].tolist()
  results['MFCC_12']=F[19].tolist()
  results['MFCC_13']=F[20].tolist()
  results['chroma_vector_1']=F[21].tolist()
  results['chroma_vector_2']=F[22].tolist()
  results['chroma_vector_3']=F[23].tolist()
  results['chroma_vector_4']=F[24].tolist()
  results['chroma_vector_5']=F[25].tolist()
  results['chroma_vector_6']=F[26].tolist()
  results['chroma_vector_7']=F[27].tolist()
  results['chroma_vector_8']=F[28].tolist()
  results['chroma_vector_9']=F[29].tolist()
  results['chroma_vector_10']=F[30].tolist()
  results['chroma_vector_11']=F[31].tolist()
  results['chroma_vector_12']=F[32].tolist()
  results['chroma_deviation'] = F[33].tolist()

  return results
