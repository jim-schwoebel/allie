import os

def augment_silence(filename):
	new_filename=filename[0:-4]+'_trimmed.wav'
	command='sox %s %s silence -l 1 0.1 1'%(filename, new_filename)+"% -1 2.0 1%"
	os.system(command)