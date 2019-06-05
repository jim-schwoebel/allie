import shutil, os, random
from pydub import AudioSegment
try:
	os.mkdir('noise')
except:
	shutil.rmtree('noise')
	os.mkdir('noise')
def extract_noise(filename, length):
	song = AudioSegment.from_mp3(filename)
	first = song[100:100+length]
	first.export(filename[0:-4]+'_noise.mp3')
	shutil.move(os.getcwd()+'/'+filename[0:-4]+'_noise.mp3', os.getcwd()+'/noise/'+filename[0:-4]+'_noise.mp3')
listdir=os.listdir()
mp3files=list()
for i in range(len(listdir)):
	if listdir[i][-4:]=='.mp3':
		mp3files.append(listdir[i])
random.shuffle(mp3files)
for i in range(len(mp3files)):
	extract_noise(mp3files[i],300)
	if i == 100:
		break
os.chdir('noise')
listdir=os.listdir()
for i in range(len(listdir)):
	if listdir[i][-4:]=='.mp3':
		os.system('play %s'%(listdir[i]))
		remove=input('should remove? type y to remove')
		if remove=='y':
			os.remove(listdir[i])
