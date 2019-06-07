import parselmouth, sys, os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def specimage2_featurize(wavfile, cur_dir, haar_dir):
	sns.set() # Use seaborn's default style to make attractive graphs
	# Plot nice figures using Python's "standard" matplotlib library
	snd = parselmouth.Sound(wavfile)

	def draw_pitch(pitch):
	    # Extract selected pitch contour, and
	    # replace unvoiced samples by NaN to not plot
	    pitch_values = pitch.selected_array['frequency']
	    pitch_values[pitch_values==0] = np.nan
	    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
	    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
	    plt.grid(False)
	    plt.ylim(0, pitch.ceiling)
	    plt.ylabel("fundamental frequency [Hz]")

	def draw_spectrogram(spectrogram, dynamic_range=70):
	    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
	    sg_db = 10 * np.log10(spectrogram.values)
	    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
	    plt.ylim([spectrogram.ymin, spectrogram.ymax])
	    plt.xlabel("time [s]")
	    plt.ylabel("frequency [Hz]")

	def draw_intensity(intensity):
	    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
	    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
	    plt.grid(False)
	    plt.ylim(0)
	    plt.ylabel("intensity [dB]")
	    
	pitch = snd.to_pitch()
	# If desired, pre-emphasize the sound fragment before calculating the spectrogram
	pre_emphasized_snd = snd.copy()
	pre_emphasized_snd.pre_emphasize()
	spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
	plt.figure()
	draw_spectrogram(spectrogram)
	plt.twinx()
	draw_pitch(pitch)
	plt.xlim([snd.xmin, snd.xmax])
	# plt.show() # or plt.savefig("spectrogram_0.03.pdf")
	imgfile=wavfile[0:-4]+'.png'
	plt.savefig(imgfile)

	features, labels=imf.image_featurize(cur_dir, haar_dir, imgfile)
	# remove temporary image file 
	os.remove(wavfile[0:-4]+'.png')

	return filename