import os
import soundfile as sf

def clean_getfirst3secs(audiofile):
	data, samplerate = sf.read(audiofile)
	os.remove(audiofile)
	data2=data[0:samplerate*3]
	sf.write(audiofile,data2, samplerate)