import os
import soundfile as sf

listdir=os.listdir()
mp3files=list()
for i in range(len(listdir)):
	if listdir[i].endswith('.mp3'):
		mp3files.append(listdir[i])

for i in range(len(mp3files)):
	os.system('ffmpeg -i %s %s'%(mp3files[i], mp3files[i][0:-4]+'.wav'))
	os.remove(mp3files[i])
