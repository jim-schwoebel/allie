import os

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

curdir=os.getcwd()
prevdir=prev_dir(curdir)
train_dir=prevdir+'/train_dir'
os.chdir(train_dir)
listdir=os.listdir()
csvfiles=list()
for i in range(len(listdir)):
	if listdir[i].endswith('.csv'):
		csvfiles.append(listdir[i])

# now train models
os.chdir(curdir)

for i in range(len(csvfiles)):
	os.system('python3 model.py r %s %s'%(csvfiles[i], csvfiles[i][0:-4]))
