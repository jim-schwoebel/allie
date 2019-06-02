from __future__ import division
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
import helpers.sa.features.signal as ef
import sys, json, os

def sa_featurize(filename):

	print('calculating frequencies...')

	# convert to mono (and a temp file 'back.wav')
	tempfilename='themostemporaryfilename_ever_2301394134134.wav'
	if tempfilename in os.listdir():
		os.remove(tempfilename)
	os.system('ffmpeg -i %s -ac 1 %s'%(filename, tempfilename))
	jsonfilename=filename[0:-4]+'.json'
	fs, data = wav.read(tempfilename)
	os.remove(tempfilename)

	def get_f0(data, fs):
		'''
		good for classifying genders 
		'''
		f0=ef.get_F_0(data, fs, time_step=0.0, min_pitch=75, max_pitch=600, max_num_cands=15, silence_thres=0.03, voicing_thres=0.45, octave_cost=0.01, octave_jump_cost=0.35, voiced_unvoiced_cost=0.14, accurate=False, pulse=False)
		return f0

	def get_jitter(data,fs):
		'''
		Jitter is the measurement of random pertubations in period length. For most 
	    accurate jitter measurements, calculations are typically performed on long 
	    sustained vowels.
	    '''
		jitter=ef.get_Jitter(data, fs, period_floor = .0001, period_ceiling = .02, max_period_factor = 1.3 )
		return jitter

	def get_pulses(data,fs):
		'''
		Computes glottal pulses of a signal.
	    This algorithm relies on the voiced/unvoiced decisions and fundamental 
	    frequencies, calculated for each voiced frame by get_F_0. For every voiced 
	    interval, a list of points is created by finding the initial point 
	    :math:`t_1`, which is the absolute extremum ( or the maximum/minimum, 
	    depending on your include_max and include_min parameters ) of the amplitude 
	    of the sound in the interval 
	    '''
		pulses=ef.get_Pulses(data, fs, min_pitch = 75, max_pitch = 600, include_max = False, include_min = True )
		return pulses 

	def get_hnr(data,fs):
		'''
		measure of hoarseness 
		'''
		hnr=ef.get_HNR(data, fs, time_step = 0, min_pitch = 75, silence_threshold = .1, periods_per_window = 4.5 )
		return hnr 

	# get statistical features in numpy
	def stats(matrix):
	    mean=np.mean(matrix)
	    std=np.std(matrix)
	    maxv=np.amax(matrix)
	    minv=np.amin(matrix)
	    median=np.median(matrix)

	    output=np.array([mean,std,maxv,minv,median])
	    
	    return output

	# get labels for later 
	def stats_labels(label, sample_list):
	    mean=label+'_mean'
	    std=label+'_std'
	    maxv=label+'_maxv'
	    minv=label+'_minv'
	    median=label+'_median'
	    sample_list.append(mean)
	    sample_list.append(std)
	    sample_list.append(maxv)
	    sample_list.append(minv)
	    sample_list.append(median)

	    return sample_list

	labels=list()

	# calculate HNR 
	hnr=get_hnr(data,fs)
	labels.append('hnr')

	# get pulses and calculate stats from pulses 
	pulses=get_pulses(data,fs)
	pulses=stats(pulses)
	features=np.append(hnr, pulses)
	labels=stats_labels('pulses', labels)

	# get jitter 
	jitter=get_jitter(data,fs)
	features=np.append(features,np.array(list(jitter.values())))
	jitter_labels=list(jitter)
	new_jitter_labels=list()
	for i in range(len(jitter_labels)):
		new_jitter_labels.append('jitter_'+jitter_labels[i])

	labels=labels+new_jitter_labels

	# get fundamental frequency
	f0=get_f0(data,fs)
	features=np.append(features,f0[0])
	labels.append('f0')

	return features, labels 

