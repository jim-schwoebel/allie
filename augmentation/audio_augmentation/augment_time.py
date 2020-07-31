import os, librosa

def augment_time(filename):
	'''
	stretches files by 0.5x, 1.5x, and 2x.
	'''
	basefile=filename[0:-4]
	filenames=list()

	y, sr = librosa.load(filename)

	y_fast = librosa.effects.time_stretch(y, 1.5)
	librosa.output.write_wav(basefile+'_stretch_0.wav', y_fast, sr)

	# y_fast_2 = librosa.effects.time_stretch(y, 1.5)
	# librosa.output.write_wav(basefile+'_stretch_1.wav', y, sr)
	# filenames.append(basefile+'_stretch_1.wav')

	y_slow = librosa.effects.time_stretch(y, 0.75)
	librosa.output.write_wav(basefile+'_stretch_2.wav', y_slow, sr)

	# y_slow_2 = librosa.effects.time_stretch(y, 0.25)
	# librosa.output.write_wav(basefile+'_stretch_3.wav', y, sr)
	# filenames.append(basefile+'_stretch_3.wav')

	return filenames 