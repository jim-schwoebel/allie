# import required modules 
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import sys,json, os 
import numpy as np

def stats(matrix):
    mean=np.mean(matrix)
    std=np.std(matrix)
    maxv=np.amax(matrix)
    minv=np.amin(matrix)
    median=np.median(matrix)
    output=np.array([mean,std,maxv,minv,median])
    return output

def convert_mono(filename):
    mono=filename[0:-4]+'_mono.wav'
    os.system('ffmpeg -i %s -ac 1 %s'%(filename,mono))
    return mono

filename=sys.argv[1]
print(filename)
mono=convert_mono(filename)
[Fs, x] = audioBasicIO.readAudioFile(mono)
features, labels= audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)

new_features=list()
new_labels=list()

for i in range(len(features)):
    tfeatures=stats(features[i])
    new_features=np.append(new_features,tfeatures)
    new_labels.append('mean '+labels[i])
    new_labels.append('std '+labels[i])
    new_labels.append('max '+labels[i])
    new_labels.append('min '+labels[i])
    new_labels.append('median '+labels[i])

os.remove(mono)
os.remove(filename)

data={'features': new_features.tolist(),
	  'labels': new_labels}
jsonfile=open(filename[0:-4]+'.json','w')
json.dump(data,jsonfile)
jsonfile.close() 

