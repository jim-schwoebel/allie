'''
Quickly generate some sample audio data from a GitHub repository.
'''

import os, shutil

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

listdir=os.listdir()
if 'sample_voice_data' not in listdir:
	os.system('git clone git@github.com:jim-schwoebel/sample_voice_data.git')
else:
	pass

cur_dir=os.getcwd()
base_dir=prev_dir(cur_dir)
train_dir=base_dir+'/train_dir'

try:
	shutil.copytree(cur_dir+'/sample_voice_data/males',train_dir+'/males')
except:
	shutil.rmtree(train_dir+'/males')
	shutil.copytree(cur_dir+'/sample_voice_data/males',train_dir+'/males')
try:
	shutil.copytree(cur_dir+'/sample_voice_data/females',train_dir+'/females')
except:
	shutil.rmtree(train_dir+'/females')
	shutil.copytree(cur_dir+'/sample_voice_data/females',train_dir+'/females')