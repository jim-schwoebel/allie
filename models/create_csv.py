import os, sys, time
import pandas as pd

def detect_files(listdir, directory):
	audios=list()
	images=list()
	texts=list()
	videos=list()
	for i in range(len(listdir)):
		if listdir[i].endswith('.wav'):
			audios.append(directory+'/'+listdir[i])
		elif listdir[i].endswith('.png'):
			images.append(directory+'/'+listdir[i])
		elif listdir[i].endswith('.txt'):
			texts.append(directory+'/'+listdir[i])
		elif listdir[i].endswith('.mp4'):
			videos.append(directory+'/'+listdir[i])

	array_=[len(audios), len(images), len(texts), len(videos)]
	maxval=max(array_)
	labels=list()
	label=directory.split('/')[-1]
	for i in range(maxval):
		labels.append(label)

	ind=array_.index(maxval)
	if ind == 0:
		data={'data': audios}
	elif ind == 1:
		data={'data': images}
	elif ind == 2: 
		data={'data': texts}
	elif ind == 3:
		data={'data': videos}
	
	data['labels']=labels

	return data, maxval

def get_ind(data, start):
	array_=[]
	for i in range(len(data)):
		array_.append(i+start)
	return array_

def get_dataframe(directory, start):

	os.chdir(directory)
	listdir=os.listdir()
	start=0
	data, start=detect_files(listdir, directory)
	data=pd.DataFrame(data)
	ind1=get_ind(data, 0)
	data=pd.DataFrame(data, index=ind1)

	return data, start 

curdir=os.getcwd()

directories=list()
i=1
while True:
	try:
		directories.append(sys.argv[i])
		i=i+1
	except:
		break

start=0
datas=list()
for i in range(len(directories)):
	data, start=get_dataframe(directories[i], start)
	datas.append(data)

for i in range(len(datas)-1):
	data=pd.concat([data, datas[i]])

# combine spreadsheets
os.chdir(curdir)
data.to_csv('output.csv', index=False)