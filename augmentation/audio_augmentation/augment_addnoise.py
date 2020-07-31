import os, uuid, random, math
from pydub import AudioSegment

def augment_addnoise(filename,curdir, noisedir):
	if filename[-4:]=='.wav':
		audioseg=AudioSegment.from_wav(filename)
	elif filename[-4:]=='.mp3':
		audioseg=AudioSegment.from_mp3(filename)
	hostdir=os.getcwd()
	os.chdir(curdir)
	os.chdir(noisedir)
	listdir=os.listdir()
	if 'noise.wav' in listdir:
		os.remove('noise.wav')
	mp3files=list()
	for i in range(len(listdir)):
		if listdir[i][-4:]=='.mp3':
			mp3files.append(listdir[i])
	noise=random.choice(mp3files)
	# add noise to the regular file 
	noise_seg = AudioSegment.from_mp3(noise)
	# find number of noise segments needed
	cuts=math.floor(len(audioseg)/len(noise_seg))
	noise_seg_2=noise_seg * cuts
	noise_seg_3=noise_seg[:(len(audioseg)-len(noise_seg_2))] 
	noise_seg_4=noise_seg_2 + noise_seg_3
	os.chdir(hostdir)
	print(len(noise_seg_4))
	print(len(audioseg))
	noise_seg_4.export("noise.wav", format="wav")
	# now combine audio file and noise file 
	os.system('ffmpeg -i %s -i %s -filter_complex "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]" -map "[a]" %s'%(filename, 'noise.wav',filename[0:-4]+'_noise.wav'))
	os.remove('noise.wav')