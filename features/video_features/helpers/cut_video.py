import os, librosa, math, shutil

def cut_video(video, splitlength):
	audio=video[0:-4]+'.wav'
	foldername=video[0:-4]
	os.system('ffmpeg -i %s %s'%(video, audio))
	y, sr=librosa.core.load(audio)
	duration=librosa.core.get_duration(y,sr)
	splits=math.floor(duration/10)
	count=0
	curdir=os.getcwd()
	
	try:
		os.mkdir(foldername)
		os.chdir(foldername)
	except:
		shutil.rmtree(foldername)
		os.mkdir(foldername)
		os.chdir(foldername)

	shutil.copy(curdir+'/'+video, curdir+'/'+foldername+'/'+video)
	
	for i in range(splits):
		os.system('ffmpeg -i %s -ss %s -t %s %s.mp4'%(video, str(count), str(count+10), video[0:-4]+'_'+str(i)+'.mp4'))
		count=count+10

cut_video('test.mp4',10)