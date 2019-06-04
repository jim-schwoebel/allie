import os
from tqdm import tqdm


os.chdir('/Users/jimschwoebel/desktop/deletejson')
hostdir=os.getcwd()
listdir=os.listdir()
folders=list()
for i in range(len(listdir)):
	if listdir[i].find('.') < 0:
		folders.append(listdir[i])

for i in tqdm(range(len(folders))):
	os.chdir(folders[i])
	listdir=os.listdir()
	for j in range(len(listdir)):
		if listdir[j][-5:]=='.json':
			os.remove(listdir[j])
	os.chdir(hostdir)



