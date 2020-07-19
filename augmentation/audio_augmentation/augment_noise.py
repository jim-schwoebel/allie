import os

def augment_noise(filename):

	'''
	following remove_noise.py from voicebook.
	'''

	#now use sox to denoise using the noise profile
	data, samplerate =sf.read(filename)
	duration=data/samplerate
	first_data=samplerate/10
	filter_data=list()
	for i in range(int(first_data)):
	    filter_data.append(data[i])
	noisefile='noiseprof.wav'
	sf.write(noisefile, filter_data, samplerate)
	os.system('sox %s -n noiseprof noise.prof'%(noisefile))
	filename2='tempfile.wav'
	filename3='tempfile2.wav'
	noisereduction="sox %s %s noisered noise.prof 0.21 "%(filename,filename2)
	command=noisereduction
	#run command 
	os.system(command)
	print(command)
	#reduce silence again
	#os.system(silenceremove)
	#print(silenceremove)
	#rename and remove files 
	os.rename(filename2,filename[0:-4]+'_noise_remove.wav')
	#os.remove(filename2)
	os.remove(noisefile)
	os.remove('noise.prof')