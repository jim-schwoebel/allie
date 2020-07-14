import os
import soundfile as sf

listdir=os.listdir()
wavfiles=list()
for i in range(len(listdir)):
	if listdir[i].endswith('.wav'):
		wavfiles.append(listdir[i])

os.mkdir('3secs')

curdir=os.getcwd()

for i in range(len(wavfiles)):
	os.chdir(curdir)
	data, samplerate = sf.read(wavfiles[i])
	data2=data[0:samplerate*3]
	os.chdir('3secs')
	sf.write(wavfiles[i],data2, samplerate)
