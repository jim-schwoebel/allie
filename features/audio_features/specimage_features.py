import os, sys
import helpers.audio_plot as ap 

def prev_dir(directory):
	g=directory.split('/')
	# print(g)
	lastdir=g[len(g)-1]
	i1=directory.find(lastdir)
	directory=directory[0:i1]
	return directory

# import to get image feature script 
directory=os.getcwd()
prevdir=prev_dir(directory)
sys.path.append(prevdir+'/image_features')
haar_dir=prevdir+'image_features/helpers/haarcascades'
import image_features as imf
os.chdir(directory)

def specimage_featurize(wavfile, cur_dir, haar_dir):
	# create temporary image 
	imgfile=ap.plot_spectrogram(wavfile)
	features, labels=imf.image_featurize(cur_dir, haar_dir, imgfile)
	# remove temporary image file 
	os.remove(wavfile[0:-4]+'.png')

	return features, labels 