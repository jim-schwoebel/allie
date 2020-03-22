import seaborn, pysptk, matplotlib
import numpy as np
from scipy.io import wavfile

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

def pysptk_featurize(audiofile):
	labels=list()
	features=list()
	fs, x = wavfile.read(audiofile)

	f0_swipe = pysptk.swipe(x.astype(np.float64), fs=fs, hopsize=80, min=60, max=200, otype="f0")
	features=features+stats(f0_swipe)
	labels=stats_labels('f0_swipe',labels)

	f0_rapt = pysptk.rapt(x.astype(np.float32), fs=fs, hopsize=80, min=60, max=200, otype="f0")
	features=features+stats(f0_rapt)
	labels=stats_labels('f0_rapt',labels)

	mgc = pysptk.mgcep(xw, 20, 0.0, 0.0)
	features=features+stats(mgc)
	labels=stats_labels('mel-spectrum envelope',labels)

	return features, labels

features, labels = pysptk_featurize('test.wav')
print(features)
print(labels)