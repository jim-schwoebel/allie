import os,uuid, sys

directory=sys.argv[1]
os.chdir(directory)
listdir=os.listdir()

for i in range(len(listdir)):
	if listdir[i].endswith('.wav'):
		newname=str(uuid.uuid4())
		os.rename(listdir[i],newname+'.wav')
		if listdir[i][0:-4]+'.json' in listdir:
			os.rename(listdir[i][0:-4]+'.json', newname+'.json')