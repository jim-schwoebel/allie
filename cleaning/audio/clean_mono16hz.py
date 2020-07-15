import os
import soundfile as sf

def clean_mono16hz(audiofile):
	# replace wavfile with a version that is 16000 Hz mono audio
	if audiofile.endswith('.wav'):
		os.system('ffmpeg -i "%s" -ab 16k "%s" -y'%(audiofile,audiofile))
	elif audiofile.endswith('.mp3'):
		os.system('ffmpeg -i "%s" -ab 16k "%s" -y'%(audiofile,audiofile[0:-4]+'.wav'))