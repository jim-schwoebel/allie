'''
PLDA implementation from
https://github.com/RaviSoji/plda/blob/master/mnist_demo/mnist_demo.ipynb
'''

import os, sys, plda, pickle 
import numpy as np
import matplotlib.pyplot as plt

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
	countvalues=counts.values()
	maxvalues=max(countvalues)
	maxind=countvalues.index(maxvalues)
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
default_csv_features=settings['default_csv_features']

# prepare training and testing data (should have been already featurized) - # of classes/folders
os.chdir(prev_dir+'/train_dir')
data_dir=os.getcwd()
listdir=os.listdir()
folders=get_folders(listdir)

# now assess folders by content type 
data=dict()
for i in range(len(folders)):
	os.chdir(folder)
	listdir=os.listdir()
	filetype=classifyfolder(listdir)
	data[folders[i]]=filetype 
	os.chdir(data_dir)

# now ask user what type of problem they are trying to solve 
problemtype=input('what problem are you solving? (1-audio, 2-text, 3-image, 4-video, 5-csv)')
while problemtype not in ['1','2','3','4','5']:
	print('answer not recognized...')
	problemtype=input('what problem are you solving? (1-audio, 2-text, 3-image, 4-video, 5-csv)')

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

print('\n OK cool, we got you modeling %s files'%(problemtype))
print('\n')
print('these are the available classes:')


# fit a PLDA to a dataset of two classes. 
training_data = np.load('mnist_demo/mnist_data/mnist_train_images.npy')
training_labels = np.load('mnist_demo/mnist_data/mnist_train_labels.npy')
testing_data = np.load('mnist_demo/mnist_data/mnist_test_images.npy')
testing_labels = np.load('mnist_demo/mnist_data/mnist_test_labels.npy')

print(training_data.shape, training_labels.shape)
print(testing_data.shape, testing_labels.shape)

# optimize number of principal components in terms of accuracy
acclist=list()
compnum=list()
for i in range(2, 190,1):
	try:
		classifier = plda.Classifier()
		numcomponents=i
		classifier.fit_model(training_data, training_labels, n_principal_components=numcomponents)
		predictions, log_p_predictions = classifier.predict(testing_data)
		accuracy=(testing_labels == predictions).mean()
		acclist.append(accuracy)
		print(accuracy)
		print(i)
	except:
		# if dimension too high, break it 
		break 

maxacc=max(acclist)
numcomponents=acclist.index(maxacc)+1 

# now retrain with proper parameters 
classifier = plda.Classifier()
classifier.fit_model(training_data, training_labels, n_principal_components=numcomponents)
predictions, log_p_predictions = classifier.predict(testing_data)
accuracy=(testing_labels == predictions).mean()
print('max acc %s with %s components'%(maxacc, numcomponents))

Psi = classifier.model.Psi
A = classifier.model.A
inv_A = classifier.model.inv_A
m = classifier.model.m

# Indices of the subspace used for classification.
relevant_U_dims = classifier.model.relevant_U_dims

# # Prior Gaussian Parameters
# classifier.model.prior_params.keys()
# # Posterior Gaussian Parameters
# classifier.model.posterior_params.keys()
# classifier.model.posterior_params[0].keys()
# # Posterior Predictive Gaussian Parameters
# classifier.model.posterior_predictive_params.keys()
# classifier.model.posterior_predictive_params[0].keys()

'''
Transforming Data to PLDA Space
There are 4 "spaces" that result from the transformations the model performs:

Data space ('D'),
Preprocessed data space ('X'),
Latent space ('U'), and
The "effective" subspace of the latent space ('U_model'), which is essentially the set of dimensions the model actually uses for prediction.
'''

U_model = classifier.model.transform(training_data, from_space='D', to_space='U_model')
print(training_data.shape)
print(U_model.shape)

D = classifier.model.transform(U_model, from_space='U_model', to_space='D')
print(U_model.shape)
print(D.shape)

# create dump of classifier 
print('saving classifier to disk')
f=open('plda_classifier.pickle','wb')
pickle.dump(classifier,f)
f.close()