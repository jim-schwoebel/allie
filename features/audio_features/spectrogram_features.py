'''
140416 features in frequency domain for spectrograms. 
Resampled down to 32000 Hz and mono signal. 
On Log mel spectrogram scale. 
'''
import os
import librosa
import numpy as np
from sklearn.decomposition import PCA

def spectrogram_featurize(file_path):
	n_fft = 1024
	sr = 32000
	mono = True
	log_spec = False
	n_mels = 128

	hop_length = 192
	fmax = None

	if mono:
		sig, sr = librosa.load(file_path, sr=sr, mono=True)
		sig = sig[np.newaxis]
	else:
		sig, sr = librosa.load(file_path, sr=sr, mono=False)

	spectrograms = []
	for y in sig:

		# compute stft
		stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True, pad_mode='reflect')

		# keep only amplitures
		stft = np.abs(stft)

		# log spectrogram 
		stft = np.log10(stft + 1)

		# apply mel filterbank
		spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)

		# keep spectrogram
		spectrograms.append(np.asarray(spectrogram))

	# np.ndarray.flatten(
	features=np.ndarray.flatten(np.asarray(spectrograms))

	# need to load a dimensionality reduction algorithm here (PCA load) to get down to 100 dimensions.

	labels=list()
	for i in range(len(features)):
		labels.append('log_spectrogram_feature_%s'%(str(i+1)))

	return features, labels 

# featurelist=list()
# os.chdir('test2')
# listdir=os.listdir()
# for i in range(len(listdir)):
# 	if listdir[i][-4:] in ['.wav', '.mp3']:
# 		features, labels = spectrogram_featurize(listdir[i])
# 		featurelist.append(features)
# 		print(len(features))

