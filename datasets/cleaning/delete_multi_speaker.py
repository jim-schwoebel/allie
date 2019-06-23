
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

print('-----------------------------')
print(' DELETING MUPLTIPLE SPEAKERS ')
print('-----------------------------')

folderpath=sys.argv[1]
print(folderpath)

model_dir=sys.argv[2]
curdir=os.getcwd()
model = keras.models.load_model(model_dir+'/models/RNN_keras2.h5')
with np.load(model_dir+'/models/scaler.npz') as data:
	mean_ = data['arr_0']
	scale_ = data['arr_1']

os.chdir(folderpath)
listdir=os.listdir()
wavfiles=get_wavfiles(listdir)
errors=list()

for i in range(len(wavfiles)):
    try:
        speaker_number=get_speakernum(wavfiles[i], model, mean_,scale_)
        if speaker_number != 1: 
            # remove files with more than 1 concurrent speaker 
            os.remove(wavfiles[i])
    except:
        print('error processing audio')
        errors.append(wavfiles[i])

print(errors)