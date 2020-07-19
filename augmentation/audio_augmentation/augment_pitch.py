import os

def augment_pitch(filename):
	'''
	takes in an audio file and outputs files normalized to 
	different pitches. This corrects for gender ane time-of-day differences.

	where gives the pitch shift as positive or negative ‘cents’ (i.e. 100ths of a semitone). 
	There are 12 semitones to an octave, so that would mean ±1200 as a parameter.
	'''
	filenames=list()

	basefile=filename[0:-4]
	# down two octave 
	# os.system('sox %s %s pitch -2400'%(filename, basefile+'_freq_0.wav'))
	# filenames.append(basefile+'_freq_0.wav')

	# down two octave 
	os.system('sox %s %s pitch -600'%(filename, basefile+'_freq_1.wav'))
	filenames.append(basefile+'_freq_1.wav')

	# up one octave 
	os.system('sox %s %s pitch 600'%(filename, basefile+'_freq_2.wav'))
	filenames.append(basefile+'_freq_2.wav')

	# up two octaves 
	# os.system('sox %s %s pitch 2400'%(filename, basefile+'_freq_3.wav'))
	# filenames.append(basefile+'_freq_3.wav')

	return filenames 
