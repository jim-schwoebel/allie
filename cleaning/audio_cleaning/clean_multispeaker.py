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


/  __ \ |                (_)              / _ \ | ___ \_   _|  _ 
| /  \/ | ___  __ _ _ __  _ _ __   __ _  / /_\ \| |_/ / | |   (_)
| |   | |/ _ \/ _` | '_ \| | '_ \ / _` | |  _  ||  __/  | |      
| \__/\ |  __/ (_| | | | | | | | | (_| | | | | || |    _| |_   _ 
 \____/_|\___|\__,_|_| |_|_|_| |_|\__, | \_| |_/\_|    \___/  (_)
								   __/ |                         
								  |___/                          
  ___            _ _       
 / _ \          | (_)      
/ /_\ \_   _  __| |_  ___  
|  _  | | | |/ _` | |/ _ \ 
| | | | |_| | (_| | | (_) |
\_| |_/\__,_|\__,_|_|\___/ 
						   

This cleaning script converts deletes all audio files in a folder that have multiple
speakers as determined by a deep learning model. Note that this works good on small files
under 10 seconds in length but can be inaccurate for longer length audio files.

This cleaning script is enabled if default_audio_cleaners=['clean_multispeaker'] 
'''
import numpy as np
import soundfile as sf
import argparse, os, keras, sklearn, librosa, sys

def get_speakernum(filename, model, mean_, scale_):
	'''
	taken from https://github.com/faroit/CountNet
	(research paper - https://arxiv.org/abs/1712.04555).

	Note this is the number of concurrent speakers (in parallel), 
	and can be used to detect ambient noise. 

	Note also that it may be better to break up speech into 5 second
	segments here for better accuracy, as the model is biased for this
	particular case. 
	'''
	print(filename)
	eps = np.finfo(np.float).eps
	# load standardisation parameters
	scaler = sklearn.preprocessing.StandardScaler()
	scaler.mean_=mean_
	scaler.scale_=scale_
	# compute audio
	audio, rate = sf.read(filename, always_2d=True)
	# downmix to mono
	audio = np.mean(audio, axis=1)
	# compute STFT
	X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T
	# apply standardization
	X = scaler.transform(X)
	# cut to input shape length (500 frames x 201 STFT bins)
	X = X[:model.input_shape[1], :]
	# apply normalization
	Theta = np.linalg.norm(X, axis=1) + eps
	X /= np.mean(Theta)
	# add sample dimension
	Xs = X[np.newaxis, ...]
	# predict output
	ys = model.predict(Xs, verbose=0)
	print("Speaker Count Estimate: ", np.argmax(ys, axis=1)[0])

	return np.argmax(ys, axis=1)[0]

def get_wavfiles(listdir):
	wavfiles=list()
	for i in range(len(listdir)):
		if listdir[i][-4:]=='.wav':
			wavfiles.append(listdir[i])

	return wavfiles 

def clean_multispeaker(audiofile,modeldir):
	curdir=os.getcwd()
	model = keras.models.load_model(modeldir+'/RNN_keras2.h5')
	with np.load(modeldir+'/scaler.npz') as data:
		mean_ = data['arr_0']
		scale_ = data['arr_1']

	try:
		speaker_number=get_speakernum(audiofile, model, mean_,scale_)
		print(speaker_number)
		if speaker_number > 1: 
			# remove files with more than 1 concurrent speaker 
			os.remove(audiofile)
			return []
		else:
			return [audiofile]
	except:
		print('error')
		os.remove(audiofile)
		return []