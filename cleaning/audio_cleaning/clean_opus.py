import os, shutil

def clean_opus(filename, opusdir):

	filenames=list()

	#########################
	# lossy codec - .opus 
	#########################
	curdir=os.getcwd()
	newfile=filename[0:-4]+'.opus'

	# copy file to opus encoding folder 
	shutil.copy(curdir+'/'+filename, opusdir+'/'+filename)
	os.chdir(opusdir)
	print(os.getcwd())
	# encode with opus codec 
	os.system('opusenc %s %s'%(filename,newfile))
	os.remove(filename)
	filename=filename[0:-4]+'_opus.wav'
	os.system('opusdec %s %s'%(newfile, filename))
	os.remove(newfile)
	# delete .wav file in original dir 
	shutil.copy(opusdir+'/'+filename, curdir+'/'+filename)
	os.remove(filename)
	os.chdir(curdir)
	os.remove(newfile[0:-5]+'.wav')
