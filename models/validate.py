import os,json
from tqdm import tqdm
import time
def prev_dir(directory):
	g=directory.split('/')
	dir_=''
	for i in range(len(g)):
		if i != len(g)-1:
			if i==0:
				dir_=dir_+g[i]
			else:
				dir_=dir_+'/'+g[i]
	# print(dir_)
	return dir_

os.chdir(prev_dir(os.getcwd())+'/load_dir/')
listdir=os.listdir()
jsonfiles=list()

for i in range(len(listdir)):
    if listdir[i].endswith('.json'):
            jsonfiles.append(listdir[i])

predictions=list()
for i in tqdm(range(len(jsonfiles))):
	try:
		g=json.load(open(jsonfiles[i]))
		models=list(g['models']['audio'])
		predictions=predictions+models
	except:
		print('error')

print('females')
print(predictions.count('female'))
print('males')
print(predictions.count('male'))