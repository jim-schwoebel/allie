'''
Featurizes a master spreadsheet of files with all the default featurizers.

This was inspired by the D3M schema by MIT Data lab. More info about this schema 
can be found @ https://github.com/mitll/d3m-schema/blob/master/documentation/datasetSchema.md
'''

#########################################
## 			IMPORT STATEMENTS    	   ##
#########################################

import pandas as pd
import os, json, uuid, shutil, time
from sklearn import preprocessing
import pandas as pd

#########################################
## 			HELPER FUNCTIONS		   ##
#########################################

def most_common(lst):
	'''
	get most common item in a list
	'''
	return max(set(lst), key=lst.count)

def prev_dir(directory):
	'''
	Get previous directory from a host directory.
	'''
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

def element_featurize(sampletype, default_features, filepaths, directory):

	# make a temporary folder and copy all featurized files to it
	folder='%s-features-'%(sampletype)+str(uuid.uuid1())
	os.mkdir(folder)
	for i in range(len(filepaths)):
		shutil.copy(filepaths[i], directory+'/'+folder)

	# featurize the files in the folder 
	os.chdir(basedir+'/features/%s_features/'%(sampletype))
	os.system('python3 featurize.py %s'%(directory+'/'+folder))

	# get lists for outputting later
	features=list()
	labels=list()

	# go through all featurized .JSON files and read them and establish a feature array
	for i in range(len(filepaths)):
		jsonfile=filepaths[i].split('/')[-1][0:-4]+'.json'
		g=json.load(open(directory+'/'+folder+'/'+jsonfile))
		feature=[]
		label=[]
		for j in range(len(default_features)):
			array_=g['features'][sampletype][default_features[j]]
			feature=feature+array_['features']
			label=label+array_['labels']			
			features.append(feature)
			labels.append(label)
	
	# remove the temporary directory
	os.chdir(directory)
	shutil.rmtree(folder)

	return features, labels

def text_featurize_columns(filepaths, directory, settings, basedir):
	'''
	Get text features using default_text featurizer
	'''
	default_features=settings['default_text_features']
	print(default_features)
	features, labels = element_featurize('text', default_features, filepaths, directory)
	
	return features, labels

def audio_featurize_columns(filepaths, directory, settings, basedir):
	'''
	get audio features using default_audio_featurizer
	'''
	features=list()
	labels=list()
	default_features=settings['default_audio_features']
	features, labels = element_featurize('audio', default_features, filepaths, directory)

	return features, labels

def image_featurize_columns(filepaths, directory, settings, basedir):
	'''
	get image features using default_image_featuerizer
	'''
	features=list()
	labels=list()
	default_features=settings['default_image_features']

	features, labels = element_featurize('image', default_features, filepaths, directory)

	return features, labels

def video_featurize_columns(filepaths, directory, settings, basedir):
	'''
	get video features using default_video_featurizer
	'''
	features=list()
	labels=list()
	default_features=settings['default_video_features']
	features, labels = element_featurize('video', default_features, filepaths, directory)

	return features, labels

def csv_featurize_columns(filepaths, directory, settings, basedir):
	'''
	get csv features using default_csv_featurizer - likely this script.
	'''
	features=list()
	labels=list()
	default_features=settings['default_csv_features']
	features, labels = element_featurize('csv', default_features, filepaths, directory)

	return features, labels

def category_featurize_columns(columns, directory, settings, basedir):
	'''
	Create numerical representations of categorical features.
	'''
	default_features=['categorical_features']
	print(default_features)
	le = preprocessing.LabelEncoder()
	le.fit(columns)
	uniquevals=set(columns)
	features_ = list(le.transform(columns))
	labels_ = list(columns)

	# feature and labels must be arrays of arrays
	features=list()
	labels=list()
	for i in range(len(features)):
		features.append([features_[i]])
		labels.append([labels_[i]])

	return features, labels

def typedtext_featurize_columns(columns, directory, settings, basedir):
	'''
	Get text features from typed text responses
	'''
	features=list()
	labels=list()
	default_features=settings['default_text_features']
	filepaths=list()
	curdir=os.getcwd()
	folder=str('temp-'+uuid.uuid1())
	os.mkdir(folder)
	os.chdir(folder)
	for i in range(len(columns)):
		file=str(uuid.uuid1())
		textfile=open(file,'w')
		textfile.write(columns[i])
		textfile.close()
		filepaths.append(os.getcwd()+'/'+file)

	os.chdir(curdir)
	features, labels = element_featurize('text', default_features, filepaths, directory)
	shutil.rmtree(folder)

	return features, labels

# create all featurizers in a master class structure
class ColumnSample:

	# base directory for moving around folders
	basedir=prev_dir(os.getcwd())

	def __init__(self, sampletype, column, directory, settings):
		self.sampletype = sampletype
		self.column = column
		self.directory = directory
		self.settings=settings
		self.basedir = basedir

	def featurize(self):

		# if an audio file in a column, need to loop through
		print(self.sampletype)
		
		if self.sampletype == 'audio':
			features_, labels = audio_featurize_columns(self.column, self.directory, self.settings, self.basedir)
		elif self.sampletype == 'text':
			features_, labels = text_featurize_columns(self.column, self.directory, self.settings, self.basedir)
		elif self.sampletype == 'image':
			features_, labels = image_featurize_columns(self.column, self.directory, self.settings, self.basedir)
		elif self.sampletype == 'video':
			features_, labels = video_featurize_columns(self.column, self.directory, self.settings, self.basedir)
		elif self.sampletype == 'csv':
			features_, labels = csv_featurize_columns(self.column, self.directory, self.settings, self.basedir)
		elif self.sampletype == 'categorical':
			features_, labels = category_featurize_columns(self.column, self.directory, self.settings, self.basedir)
		elif self.sampletype == 'typedtext':
			features_, labels = typedtext_featurize_columns(self.column, self.directory, self.settings, self.basedir)
		
		self.features = features_
		self.labels = labels

def csv_featurize(csvfile, settings):
	# look for each column header and classify it accordingly
	if csvfile.endswith('.csv'):
		data=pd.read_csv(csvfile)
		columns=list(data)
		coltypes=list()
		datatype=list()

		for i in range(len(columns)):
			# look at filetype extension in each column
			coldata=data[columns[i]]
			sampletypes=list()
			for j in range(len(coldata)):
				if coldata[j].endswith('.wav'):
					sampletypes.append('audio')
				elif coldata[j].endswith('.txt'):
					sampletypes.append('text')
				elif coldata[j].endswith('.png'):
					sampletypes.append('image')
				elif coldata[j].endswith('.mp4'):
					sampletypes.append('video')
				else:
					# find out whether column is numerical or categorical
					if coldata[j].endswith('.csv'):
						sampletypes.append('csv')
					else:
						try:
							values=float(coldata[j])
							sampletypes.append('numerical')
						except:
							sampletypes.append('other')

			coltype=most_common(sampletypes)

			# correct the other category if needed
			if coltype == 'other':
				if coltype.endswith('.csv'):
					coltype='csv'
				elif len(set(list(coldata))) < 10:
					coltype='categorical'
				else:
					# if less than 5 unique answers then we can interpret this as text input
					coltype='typedtext'

			# now append all the columsn together
			coltypes.append(coltype)
		
		# datatypes found
		datatypes=list(set(coltypes))
		print('Data types found: %s'%(str(datatypes)))
		headers = dict(zip(columns, coltypes))
		# now go through and featurize according to the headers
		# featurize 'audio'
		curdir=os.getcwd()
		new_column_labels=list()
		new_column_values=list()
		lengths=list()
		for i in range(len(columns)):
			# get column types and featurize each sample
			sample=ColumnSample(coltypes[i], data[columns[i]], curdir, settings)
			# get back features and labels
			sample.featurize()
			features=sample.features
			labels=sample.labels
			lengths.append(len(features))
			new_column_values.append(features)
			new_column_labels.append(labels)

		print(lengths)
		 #print(new_column_labels)
		# print(new_column_values)

		return new_column_labels, new_column_values, headers, columns, data

	else:
		print('file cannot be read, as it does not end with .CSV extension!')
		headers=''
		return headers

#########################################
## 				MAIN SCRIPT		       ##
#########################################
curdir=os.getcwd()
basedir=prev_dir(os.getcwd())
os.chdir(basedir)
settings=json.load(open('settings.json'))
os.chdir(curdir)
new_column_labels, new_column_values, headers, old_column_labels, old_column_values=csv_featurize('test.csv', settings)
