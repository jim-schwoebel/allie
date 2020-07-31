import os, time, shutil
from tqdm import tqdm

listdir=os.listdir()

try:
	os.mkdir('coconut')
except:
	pass

try:
	os.mkdir('other')
except:
	pass

wavfiles=list()
for i in range(len(listdir)):
	if listdir[i].endswith('.wav'):
		wavfiles.append(listdir[i])

for i in tqdm(range(len(wavfiles))):
	wavfile=wavfiles[i]
	os.system('play %s \n'%(wavfile))
	yesorno=input('coconut? -y or -n \n')
	if yesorno == 'y':
		shutil.move(os.getcwd()+'/'+wavfile, os.getcwd()+'/coconut/'+wavfile)
	else:
		shutil.move(os.getcwd()+'/'+wavfile, os.getcwd()+'/other/'+wavfile)