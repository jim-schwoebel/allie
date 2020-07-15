import os
import soundfile as sf

def clean_mp3towav(audiofile):
	if audiofile.endswith('.mp3'):
		os.system('ffmpeg -i %s %s'%(mp3files[i], mp3files[i][0:-4]+'.wav'))
		os.remove(audiofile)