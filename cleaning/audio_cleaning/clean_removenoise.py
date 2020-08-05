'''

'''
import os, uuid

def clean_removenoise(audiofile):
	# create a noise reference (assuming linear noise)
	# following https://stackoverflow.com/questions/44159621/how-to-denoise-audio-with-sox
	# alternatives would be to use bandpass filter or other low/hf filtering techniques
	noiseaudio=str(uuid.uuid1())+'_noiseaudio.wav'
	noiseprofile=str(uuid.uuid1())+'_noise.prof'
	temp=audiofile[0:-4]+'_.wav'
	os.system('sox %s %s trim 0 0.500'%(audiofile, noiseaudio))
	os.system('sox %s -n noiseprof %s'%(noiseaudio, noiseprofile))
	os.system('sox %s %s noisered %s 0.21'%(audiofile, temp, noiseprofile))
	os.remove(audiofile)
	os.rename(temp,audiofile)
	os.remove(noiseaudio)
	os.remove(noiseprofile)

# remove_noise('test_audio.wav')
