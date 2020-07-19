import os

def augment_volume(filename):

	'''
	takes in an audio file and outputs files normalized to 
	different volumes. This corrects for microphone distance and ages.

	Note that in using ffmpeg-normalize, this mimicks real world-use.
	An alternative could be to use SoX to move up or down volume.
	'''

	def change_volume(filename, vol):
	    # rename file
	    if vol > 1:
	        new_file=filename[0:-4]+'_increase_'+str(vol)+'.wav'
	    else:
	        new_file=filename[0:-4]+'_decrease_'+str(vol)+'.wav'

	    # changes volume, vol, by input 
	    os.system('sox -v %s %s %s'%(str(vol),filename,new_file))

	    return new_file 

	filenames=list()
	basefile=filename[0:-4]
	# using peak normalization
	os.system('ffmpeg-normalize %s -nt peak -t 0 -o %s_peak_normalized.wav'%(filename, basefile))
	filenames.append('%s_peak_normalized'%(basefile))
	
	# increase volume by 2x 
	new_file=change_volume(filename, 3)
	filenames.append(new_file)
	# decrease volume by 1/2 
	new_file=change_volume(filename, 0.33)
	filenames.append(new_file)