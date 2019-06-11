'''
PLDA implementation from
https://github.com/RaviSoji/plda/blob/master/mnist_demo/mnist_demo.ipynb
'''

import os, sys, pickle, json, random, shutil
import numpy as np
import matplotlib.pyplot as plt
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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

def get_folders(listdir):
	folders=list()
	for i in range(len(listdir)):
		if listdir[i].find('.') < 0:
			folders.append(listdir[i])

	return folders 

def classifyfolder(listdir):
	filetypes=list()
	for i in range(len(listdir)):
		if listdir[i].endswith(('.mp3', '.wav')):
			filetypes.append('audio')
		elif listdir[i].endswith(('.png', '.jpg')):
			filetypes.append('image')
		elif listdir[i].endswith(('.txt')):
			filetypes.append('text')
		elif listdir[i].endswith(('.mp4', '.avi')):
			filetypes.append('video')
		elif listdir[i].endswith(('.csv')):
			filetypes.append('csv')

	counts={'audio': filetypes.count('audio'),
			'image': filetypes.count('image'),
			'text': filetypes.count('text'),
			'video': filetypes.count('video'),
			'csv': filetypes.count('.csv')}

	# get back the type of folder (main file type)
	countlist=list(counts)
	countvalues=list(counts.values())
	maxvalue=max(countvalues)
	maxind=countvalues.index(maxvalue)
	return countlist[maxind]

# load the default feature set 
cur_dir = os.getcwd()
prevdir= prev_dir(cur_dir)
sys.path.append(prevdir+'/train_dir')
settings=json.load(open(prevdir+'/settings.json'))

# get all the default feature arrays 
default_audio_features=settings['default_audio_features']
default_text_features=settings['default_text_features']
default_image_features=settings['default_image_features']
default_video_features=settings['default_video_features']
default_csv_features='csv'

# prepare training and testing data (should have been already featurized) - # of classes/folders
os.chdir(prevdir+'/train_dir')

data_dir=os.getcwd()
listdir=os.listdir()
folders=get_folders(listdir)

# now assess folders by content type 
data=dict()
for i in range(len(folders)):
	os.chdir(folders[i])
	listdir=os.listdir()
	filetype=classifyfolder(listdir)
	data[folders[i]]=filetype 
	os.chdir(data_dir)

# now ask user what type of problem they are trying to solve 
problemtype=input('what problem are you solving? (1-audio, 2-text, 3-image, 4-video, 5-csv)\n')
while problemtype not in ['1','2','3','4','5']:
	print('answer not recognized...')
	problemtype=input('what problem are you solving? (1-audio, 2-text, 3-image, 4-video, 5-csv)\n')

if problemtype=='1':
	problemtype='audio'
elif problemtype=='2':
	problemtype='text'
elif problemtype=='3':
	problemtype='image'
elif problemtype=='4':
	problemtype='video'
elif problemtype=='5':
	problemtype=='csv'

print('\n OK cool, we got you modeling %s files \n'%(problemtype))
count=0
availableclasses=list()
for i in range(len(folders)):
    if data[folders[i]]==problemtype:
    	availableclasses.append(folders[i])
    	count=count+1
classnum=input('how many classes would you like to model? (%s available) \n'%(str(count)))
print('these are the available classes: ')
print(availableclasses)
classes=list()
for i in range(int(classnum)):
	class_=input('what is class #%s \n'%(str(i+1)))
	for j in range(len(availableclasses)):
		stillavailable=list()
		if availableclasses[j] not in classes:
			stillavailable.append(availableclasses[j])
	
	if class_ == '':
		class_=stillavailable[0]

	print('available classes: %s'%(str(stillavailable)))
	classes.append(class_)

mtype=input('is this a classification (c) or regression (r) problem? \n')
while mtype not in ['c','r']:
	print('input not recognized...')
	mtype=input('is this a classification (c) or regression (r) problem? \n')

# load all default feature types 
settings=json.load(open(prevdir+'/settings.json'))
default_audio_features=settings['default_audio_features']
default_text_features=settings['default_text_features']
default_image_features=settings['default_image_features']
default_video_features=settings['default_video_features']
default_csv_features='n/a'

# now featurize each class 
data={}
for i in range(len(classes)):
	class_type=classes[i]
	if problemtype == 'audio':
		# featurize audio 
		os.chdir(prevdir+'/features/audio_features')
		default_features=default_audio_features
	elif problemtype == 'text':
		# featurize text
		os.chdir(prevdir+'/features/text_features')
		default_features=default_text_features
	elif problemtype == 'image':
		# featurize images
		os.chdir(prevdir+'/features/image_features')
		default_features=default_image_features
	elif problemtype == 'video':
		# featurize video 
		os.chdir(prevdir+'/features/video_features')
		default_features=default_video_features
	elif problemtype == '.csv':
		# featurize .CSV 
		os.chdir(prevdir+'/features/csv_features')
		default_features=default_csv_features

	os.system('python3 featurize.py %s'%(data_dir+'/'+classes[i]))
	os.chdir(data_dir+'/'+classes[i])
	# load audio features 
	listdir=os.listdir()
	feature_list=list()
	label_list=list()
	for i in range(len(listdir)):
		if listdir[i][-5:]=='.json':
			g=json.load(open(listdir[i]))
			feature_list.append(g['features'][problemtype][default_features]['features'])
			print(g['features'][problemtype][default_features]['features'])
	
	data[class_type]=feature_list

# now that we have featurizations, we can load them in and train model
os.chdir(prevdir+'/training/')

model_dir=prevdir+'/models'

jsonfile=''
for i in range(len(classes)):
    if i==0:
        jsonfile=classes[i]
    else:
        jsonfile=jsonfile+'_'+classes[i]

jsonfile=jsonfile+'.json'

#try:
g=data
alldata=list()
labels=list()
lengths=list()

# check to see all classes are same length and reshape if necessary
for i in range(len(classes)):
    class_=g[classes[i]]
    lengths.append(len(class_))

lengths=np.array(lengths)
minlength=np.amin(lengths)

# now load all the classes
for i in range(len(classes)):
    class_=g[classes[i]]
    random.shuffle(class_)

    if len(class_) > minlength:
        print('%s greater than class, equalizing...')
        class_=class_[0:minlength]

    for j in range(len(class_)):
        alldata.append(class_[i])
        labels.append(i)

os.chdir(model_dir)

alldata=np.asarray(alldata)
labels=np.asarray(labels)

# get train and test data 
X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)
if mtype in [' classification', 'c']:
    tpot=TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs=-1)
    tpotname='%s_tpotclassifier.py'%(jsonfile[0:-5])
elif mtype in ['regression','r']:
    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
    tpotname='%s_tpotregression.py'%(jsonfile[0:-5])
tpot.fit(X_train, y_train)
accuracy=tpot.score(X_test,y_test)
tpot.export(tpotname)

# export data to .json format 
data={
    'data': alldata.tolist(),
    'labels': labels.tolist(),
}

jsonfilename='%s_.json'%(tpotname[0:-3])
jsonfile=open(jsonfilename,'w')
json.dump(data,jsonfile)
jsonfile.close()

# now edit the file and run it 
g=open(tpotname).read()
g=g.replace("import numpy as np", "import numpy as np \nimport json, pickle")
g=g.replace("tpot_data = pd.read_csv(\'PATH/TO/DATA/FILE\', sep=\'COLUMN_SEPARATOR\', dtype=np.float64)","g=json.load(open('%s'))\ntpot_data=g['labels']"%(jsonfilename))
g=g.replace("features = tpot_data.drop('target', axis=1).values","features=g['data']\n")
g=g.replace("tpot_data['target'].values", "tpot_data")
g=g.replace("results = exported_pipeline.predict(testing_features)", "print('saving classifier to disk')\nf=open('%s','wb')\npickle.dump(exported_pipeline,f)\nf.close()"%(jsonfilename[0:-6]+'.pickle'))
g1=g.find('exported_pipeline = ')
g2=g.find('exported_pipeline.fit(training_features, training_target)')
modeltype=g[g1:g2]
os.remove(tpotname)
t=open(tpotname,'w')
t.write(g)
t.close()
os.system('python3 %s'%(tpotname))

# now write an accuracy label 
os.remove(jsonfilename)

jsonfilename='%s.json'%(tpotname[0:-3])
print('saving .JSON file (%s)'%(jsonfilename))
jsonfile=open(jsonfilename,'w')
if mtype in ['classification', 'c']:
    data={'sample type': problemtype,
        'feature_set':default_features,
        'model name':jsonfilename[0:-5]+'.pickle',
        'accuracy':accuracy,
        'model type':'TPOTclassification_'+modeltype,
    }
elif mtype in ['regression', 'r']:
    data={'sample type': problemtype,
        'feature_set':default_features,
        'model name':jsonfilename[0:-5]+'.pickle',
        'accuracy':accuracy,
        'model type':'TPOTregression_'+modeltype,
    }

json.dump(data,jsonfile)
jsonfile.close()

cur_dir2=os.getcwd()
try:
	os.chdir(problemtype+'_models')
except:
	os.mkdir(problemtype+'_models')
	os.chdir(problemtype+'_models')

# now move all the files over to proper model directory 
shutil.move(cur_dir2+'/'+jsonfilename, os.getcwd()+'/'+jsonfilename)
shutil.move(cur_dir2+'/'+tpotname, os.getcwd()+'/'+tpotname)
shutil.move(cur_dir2+'/'+jsonfilename[0:-6]+'.pickle', os.getcwd()+'/'+jsonfilename[0:-6]+'.pickle')
                        
#except:    
    #print('error, please put %s in %s'%(jsonfile, data_dir))
    #print('note this can be done with train_audioclassify.py script')