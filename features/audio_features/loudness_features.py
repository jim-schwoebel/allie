import soundfile as sf
import pyloudnorm as pyln
# os.system('pip3 install pyloudnorm==0.1.0')

def loudness_featurize(audiofile):
	'''
	from the docs 
	https://github.com/danilobellini/audiolazy/blob/master/examples/formants.py
	'''
	data, rate = sf.read(audiofile) # load audio (with shape (samples, channels))
	meter = pyln.Meter(rate) # create BS.1770 meter
	loudness = meter.integrated_loudness(data) # measure loudness

	# units in dB
	features=[loudness]
	labels=['Loudness']
	
	print(dict(zip(labels,features)))
	return features, labels 

# loudness_featurize('test.wav')