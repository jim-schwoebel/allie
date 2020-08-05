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
                           

This will featurize folders of audio files if the default_audio_features = ['specimage2_features']

Uses a spectrogram and features extracted from the spectrogram as feature vectors.
'''
import parselmouth, sys, os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

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
	plt.close()
	img = Image.open(wavfile[0:-4]+'.png').convert('LA')
	img.save(wavfile[0:-4]+'.png')

	features, labels=imf.image_featurize(cur_dir, haar_dir, imgfile)
	# remove temporary image file 
	# os.remove(wavfile[0:-4]+'.png')

	return features, labels 
