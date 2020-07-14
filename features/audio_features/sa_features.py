import os, librosa
from Signal_Analysis.features.signal import get_F_0, get_HNR, get_Jitter, get_Pulses
# pip3 install Signal_Analysis
import numpy as np

def signalanalysis_featurize(audiofile):
	'''
	from the docs 
	https://brookemosby.github.io/Signal_Analysis/Signal_Analysis.features.html#module-Signal_Analysis.features.signal
	'''

	y, sr = librosa.core.load(audiofile)
	duration = len(y)/sr
	print(duration)

	f0=get_F_0(y,sr)[0]
	hnr=get_HNR(y,sr)
	jitter=get_Jitter(y,sr)
	jitter_features=list(jitter.values())
	jitter_labels=list(jitter)
	pulses=get_Pulses(y,sr)
	pulses=len(pulses) / duration

	features=[f0,hnr,pulses]+jitter_features
	labels=['FundamentalFrequency','HarmonicstoNoiseRatio','PulsesPerSec']+jitter_labels

	print(dict(zip(labels,features)))

	return features, labels 

# signalanalysis_featurize('test.wav')