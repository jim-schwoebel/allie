import os
import soundfile as sf

def clean_mono16hz(audiofile):
	# replace wavfile with a version that is 16000 Hz mono audio
	if audiofile.endswith('.wav'):
		os.system('ffmpeg -i "%s" -ar 16000 -ac 1 "%s" -y'%(audiofile,audiofile[0:-4]+'_cleaned.wav'))
		os.remove(audiofile)
	elif audiofile.endswith('.mp3'):
		os.system('ffmpeg -i "%s" -ar 16000 -ac 1 "%s" -y'%(audiofile,audiofile[0:-4]+'.wav'))