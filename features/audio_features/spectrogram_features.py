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
                           

This will featurize folders of audio files if the default_audio_features = ['spectrogram_features']

140416 features in frequency domain for spectrograms.  Resampled down to 32000 Hz and mono signal. 
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

	labels=list()
	mean_features=np.mean(np.array(spectrograms), axis=2)[0]
	for i in range(len(mean_features)):
		labels.append('log_spectrogram_mean_feature_%s'%(str(i+1)))
	std_features=np.std(np.array(spectrograms), axis=2)[0]
	for i in range(len(std_features)):
		labels.append('log_spectrogram_std_feature_%s'%(str(i+1)))
	# np.ndarray.flatten
	features=np.append(mean_features, std_features)

	return features, labels 
