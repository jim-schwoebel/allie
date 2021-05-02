import os

def get_folders():
	listdir=os.listdir()
	folders=list()
	for i in range(len(listdir)):
		if listdir[i].find('.') == -1:
			folders.append(listdir[i])
	return folders

curdir=os.getcwd()
folders=get_folders()

for i in range(len(folders)):
	os.system('python3 get_stats.py %s'%(curdir+'/'+folders[i]))
