'''
Take in various sys.argv[] inputs and/or user-defined inputs and
output a trained machine learning model. 

Can featurize and model audio, image, text, video, or CSV files.

This assumes the files are separated in folders in the train_dir. 

For example, if you want to separate males and females in audio
files, you would need to have one folder named 'males' and another
folder named 'females' and specify audio file training with the
proper folder names ('male' and 'female') for the training script 
to function properly.

For automated training, you can alternatively pass through sys.argv[]
inputs as follows:

python3 model.py audio 2 c male female 

audio = audio file type 
2 = 2 classes 
c = classification (r for regression)
male = first class
female = second class [via N number of classes]
'''
###############################################################
##                  IMPORT STATEMENTS                        ##
###############################################################
import os, sys, pickle, json, random, shutil, time, itertools, uuid, datetime, uuid
from pyfiglet import Figlet
f=Figlet(font='doh')
print(f.renderText('Allie'))
f=Figlet(font='doom')
import pandas as pd
import matplotlib.pyplot as plt

###############################################################
##               CREATE HELPER FUNCTIONS                     ##
###############################################################

def most_common(lst):
	'''
	get most common item in a list
	'''
	return max(set(lst), key=lst.count)

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
			'csv': filetypes.count('csv')}

	# get back the type of folder (main file type)
	countlist=list(counts)
	countvalues=list(counts.values())
	maxvalue=max(countvalues)
	maxind=countvalues.index(maxvalue)
	return countlist[maxind]

def convert_csv(X_train, y_train, labels, mtype, classes):
	'''
	Take in a array of features and labels and output a 
	pandas DataFrame format for easy .CSV expor and for model training.

	This is important to make sure all machine learning training sessions
	use the same dataset (so they can be benchmarked appropriately).
	'''
	# from pandas merging guide https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
	feature_list=labels
	data=list()
	for i in tqdm(range(len(X_train)), desc='converting csv...'):
		
		newlist=list()
		for j in range(len(X_train[i])):
			newlist.append([X_train[i][j]])

		temp=pd.DataFrame(dict(zip(feature_list,newlist)), index=[i])
		# print(temp)
		data.append(temp)
		

	data = pd.concat(data)

	if mtype == 'c':
		data['class_']=y_train
	elif mtype == 'r':
		y_train=np.array(y_train)
		print(y_train.shape)
		for i in range(len(classes)):
			data[classes[i]]=list(y_train[:,i])

	print(data)

	data=pd.DataFrame(data, columns = list(data))
	# print this because in pretty much every case you will write the .CSV file afterwards
	print('writing csv file...')

	return data

def get_metrics(clf, problemtype, mtype, default_training_script, common_name, X_test, y_test, classes, modelname, settings, model_session, transformer_name, created_csv_files, test_data):
	'''
	get the metrics associated iwth a classification and regression problem
	and output a .JSON file with the training session.
	'''
	metrics_=dict()
	y_true=y_test

	if default_training_script not in ['autogluon', 'autokeras', 'autopytorch', 'alphapy', 'atm', 'keras', 'devol', 'ludwig', 'safe', 'neuraxle']:
		y_pred=clf.predict(X_test)
	elif default_training_script=='alphapy':
		# go to the right folder 
		curdir=os.getcwd()
		print(os.listdir())
		os.chdir(common_name+'_alphapy_session')
		alphapy_dir=os.getcwd()
		os.chdir('input')
		os.rename('test.csv', 'predict.csv')
		os.chdir(alphapy_dir)
		os.system('alphapy --predict')
		os.chdir('output')
		listdir=os.listdir()
		for k in range(len(listdir)):
			if listdir[k].startswith('predictions'):
				csvfile=listdir[k]
		y_pred=pd.read_csv(csvfile)['prediction']
		os.chdir(curdir)
	elif default_training_script == 'autogluon':
		from autogluon import TabularPrediction as task
		test_data=test_data.drop(labels=['class'],axis=1)
		y_pred=clf.predict(test_data)
	elif default_training_script == 'autokeras':
		y_pred=clf.predict(X_test).flatten()
	elif default_training_script == 'autopytorch':
		y_pred=clf.predict(X_test).flatten()
	elif default_training_script == 'atm':
		curdir=os.getcwd()
		os.chdir('atm_temp')
		data = pd.read_csv('test.csv').drop(labels=['class_'], axis=1)
		y_pred = clf.predict(data)
		os.chdir(curdir)
	elif default_training_script == 'ludwig':
		data=pd.read_csv('test.csv').drop(labels=['class_'], axis=1)
		pred=clf.predict(data)['class__predictions']
		y_pred=np.array(list(pred), dtype=np.int64)
	elif default_training_script == 'devol':
		X_test=X_test.reshape(X_test.shape+ (1,)+ (1,))
		y_pred=clf.predict_classes(X_test).flatten()
	elif default_training_script=='keras':
		if mtype == 'c':
		    y_pred=clf.predict_classes(X_test).flatten()
		elif mtype == 'r':
			y_pred=clf.predict(X_test).flatten()
	elif default_training_script=='neuraxle':
		y_pred=clf.transform(X_test)
	elif default_training_script=='safe':
		# have to make into a pandas dataframe
		test_data=pd.read_csv('test.csv').drop(columns=['class_'], axis=1)
		y_pred=clf.predict(test_data)
	
	print(y_pred)
	
	# get classification or regression metrics
	if mtype in ['c', 'classification']:
		# now get all classification metrics
		mtype='classification'
		metrics_['accuracy']=metrics.accuracy_score(y_true, y_pred)
		metrics_['balanced_accuracy']=metrics.balanced_accuracy_score(y_true, y_pred)
		try:
			metrics_['precision']=metrics.precision_score(y_true, y_pred)
		except:
			metrics_['precision']='n/a'
		try:
			metrics_['recall']=metrics.recall_score(y_true, y_pred)
		except:
			metrics_['recall']='n/a'
		try:
			metrics_['f1_score']=metrics.f1_score (y_true, y_pred, pos_label=1)
		except:
			metrics_['f1_score']='n/a'
		try:
			metrics_['f1_micro']=metrics.f1_score(y_true, y_pred, average='micro')
		except:
			metrics_['f1_micro']='n/a'
		try:
			metrics_['f1_macro']=metrics.f1_score(y_true, y_pred, average='macro')
		except:
			metrics_['f1_macro']='n/a'
		try:
			metrics_['roc_auc']=metrics.roc_auc_score(y_true, y_pred)
		except:
			metrics_['roc_auc']='n/a'
		try:
			metrics_['roc_auc_micro']=metrics.roc_auc_score(y_true, y_pred, average='micro')
		except:
			metrics_['roc_auc_micro']='n/a'
		try:
			metrics_['roc_auc_macro']=metrics.roc_auc_score(y_true, y_pred, average='macro')
		except:
			metrics_['roc_auc_micro']='n/a'
	
		metrics_['confusion_matrix']=metrics.confusion_matrix(y_true, y_pred).tolist()
		metrics_['classification_report']=metrics.classification_report(y_true, y_pred, target_names=classes)

		plot_confusion_matrix(np.array(metrics_['confusion_matrix']), classes)
		try:
			# predict_proba only works for or log loss and modified Huber loss.
			# https://stackoverflow.com/questions/47788981/sgdclassifier-with-predict-proba
			try:
				y_probas = clf.predict_proba(X_test)[:, 1]
			except:
				try:
					y_probas = clf.decision_function(X_test)[:, 1]
				except:
					print('error making y_probas')
				
			plot_roc_curve(y_test, [y_probas], [default_training_script])
		except:
			print('error plotting ROC curve')
			print('predict_proba only works for or log loss and modified Huber loss.')

	elif mtype in ['r', 'regression']:
		# now get all regression metrics
		mtype='regression'
		metrics_['mean_absolute_error'] = metrics.mean_absolute_error(y_true, y_pred)
		metrics_['mean_squared_error'] = metrics.mean_squared_error(y_true, y_pred)
		metrics_['median_absolute_error'] = metrics.median_absolute_error(y_true, y_pred)
		metrics_['r2_score'] = metrics.r2_score(y_true, y_pred)

		plot_regressor(clf, classes, X_test, y_test)

	data={'sample type': problemtype,
		  'created date': str(datetime.datetime.now()),
		  'session id': model_session,
		  'classes': classes,
	      'model type': mtype,
	      'model name': modelname, 
	      'model type': default_training_script,
	      'metrics': metrics_,
	      'settings': settings,
	      'transformer name': transformer_name,
	      'training data': created_csv_files,
	      'sample X_test': X_test[0].tolist(),
	      'sample y_test': y_test[0].tolist()}

	if modelname.endswith('.pickle'):
		jsonfilename=modelname[0:-7]+'.json'
	elif modelname.endswith('.h5'):
		jsonfilename=modelname[0:-3]+'.json'
	else:
		jsonfilename=modelname+'.json'

	jsonfile=open(jsonfilename,'w')
	json.dump(data,jsonfile)
	jsonfile.close()

def plot_roc_curve(y_test, probs, clf_names):  
	'''
	This function plots an ROC curve with the appropriate 
	list of classifiers.
	'''
	cycol = itertools.cycle('bgrcmyk')

	for i in range(len(probs)):
		print(y_test)
		print(probs[i])
		try:
			fper, tper, thresholds = roc_curve(y_test, probs[i]) 
			plt.plot(fper, tper, color=next(cycol), label=clf_names[i]+' = %s'%(str(round(metrics.auc(fper, tper), 3))))
			plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
		except:
			print('passing %s'%(clf_names[i]))

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend()
	plt.tight_layout()
	plt.savefig('roc_curve.png')
	plt.close()

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("\nNormalized confusion matrix")
	else:
		print('\nConfusion matrix, without normalization')

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	plt.savefig('confusion_matrix.png')
	plt.close()

def plot_regressor(regressor, classes, X_test, y_test):
	'''
	plot regression models with a bar chart.
	'''

	try:
		y_pred = regressor.predict(X_test)

		# plot the first 25 records
		if len(classes) == 2:
			df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
			df1 = df.head(25)
			df1.plot(kind='bar',figsize=(16,10))
			plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
			plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
			plt.tight_layout()
			plt.savefig('bar_graph_predictions.png')
			plt.close()

			# plot a straight line on the data
			plt.scatter(X_test, y_test,  color='gray')
			plt.plot(X_test, y_pred, color='red', linewidth=2)
			plt.tight_layout()
			plt.savefig('straight_line_predictions.png')
			plt.close()
		else:
			# multi-dimensional generalization 
			df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
			df1 = df.head(25)

			df1.plot(kind='bar',figsize=(10,8))
			plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
			plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
			plt.tight_layout()
			plt.savefig('bar_graph_predictions.png')
			plt.close()
	except:
		print('error plotting regressor')

###############################################################
##                    LOADING SETTINGS                       ##
###############################################################

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
create_csv=settings['create_csv']

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

###############################################################
##                  INITIALIZE CLASSES                       ##
###############################################################

# get all information from sys.argv, and if not, 
# go through asking user for the proper parameters 

try:
	problemtype=sys.argv[1]
	classnum=sys.argv[2]
	mtype=sys.argv[3]
	classes=list()
	for i in range(int(classnum)):
		classes.append(sys.argv[i+4])
except:
	# now ask user what type of problem they are trying to solve 
	mtype=input('is this a classification (c) or regression (r) problem? \n')
	while mtype not in ['c','r']:
		print('input not recognized...')
		mtype=input('is this a classification (c) or regression (r) problem? \n')

	if mtype == 'c':
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
			problemtype='csv'

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
		# get all if all (good for many classes)
		classes=list()
		if classnum=='all':
			for i in range(len(availableclasses)):
				classes.append(availableclasses[i])
		else:
					
			stillavailable=list()
			for i in range(int(classnum)):
				class_=input('what is class #%s \n'%(str(i+1)))

				while class_ not in availableclasses and class_ not in '' or class_ in classes:
					print('\n')
					print('------------------ERROR------------------')
					print('the input class does not exist (for %s files).'%(problemtype))
					print('these are the available classes: ')
					if len(stillavailable)==0:
						print(availableclasses)
					else:
						print(stillavailable)
					print('------------------------------------')
					class_=input('what is class #%s \n'%(str(i+1)))
				for j in range(len(availableclasses)):
					stillavailable=list()
					if availableclasses[j] not in classes:
						stillavailable.append(availableclasses[j])
				if class_ == '':
					class_=stillavailable[0]

				classes.append(class_)

	elif mtype =='r':
		# for regression problems we need a target column to predict / classes from a .CSV 
		problemtype='csv'
		# assumes the .CSV file is in the train dir
		os.chdir(prevdir+'/train_dir')
		listdir=os.listdir()
		csvfiles=list()
		for i in range(len(listdir)):
			if listdir[i].endswith('.csv'):
				csvfiles.append(listdir[i])

		csvfile=input('what is the name of the spreadsheet (in ./train_dir) used for prediction? \n\n available: %s\n\n'%(str(csvfiles)))
		while csvfile not in csvfiles:
			print('answer not recognized...')
			csvfile=input('what is the name of the spreadsheet (in ./train_dir) used for prediction? \n\n available: %s\n\n'%(str(csvfiles)))

		# the available classes are only the numeric columns from the spreadsheet
		data = pd.read_csv(csvfile)
		columns = list(data)
		availableclasses=list()

		for i in range(len(columns)):
			# look at filetype extension in each column
			coldata=data[columns[i]]
			sampletypes=list()
			for j in range(len(coldata)):
				try:
					values=float(coldata[j])
					sampletypes.append('numerical')
				except:
					if coldata[j].endswith('.wav'):
						sampletypes.append('audio')
					elif coldata[j].endswith('.txt'):
						sampletypes.append('text')
					elif coldata[j].endswith('.png'):
						sampletypes.append('image')
					elif coldata[j].endswith('.mp4'):
						sampletypes.append('video')
					else:
						sampletypes.append('other')

			coltype=most_common(sampletypes)

			# correct the other category if needed
			if coltype == 'other':
				# if coltype.endswith('.csv'):
					# coltype='csv'
				if len(set(list(coldata))) < 10:
					coltype='categorical'
				else:
					# if less than 5 unique answers then we can interpret this as text input
					coltype='typedtext'

			if coltype == 'numerical':
				availableclasses.append(columns[i])

		if len(availableclasses) > 0:
			classnum=input('how many classes would you like to model? (%s available) \n'%(str(len(availableclasses))))
			print('these are the available classes: %s'%(str(availableclasses)))

			classes=list()	
			stillavailable=list()
			for i in range(int(classnum)):
				class_=input('what is class #%s \n'%(str(i+1)))

				while class_ not in availableclasses and class_ not in '' or class_ in classes:
					print('\n')
					print('------------------ERROR------------------')
					print('the input class does not exist (for %s files).'%(problemtype))
					print('these are the available classes: ')
					if len(stillavailable)==0:
						print(availableclasses)
					else:
						print(stillavailable)
					print('------------------------------------')
					class_=input('what is class #%s \n'%(str(i+1)))
				for j in range(len(availableclasses)):
					stillavailable=list()
					if availableclasses[j] not in classes:
						stillavailable.append(availableclasses[j])
				if class_ == '':
					class_=stillavailable[0]

				classes.append(class_)

		else:
			print('no classes available... ending session')
			sys.exit()

	common_name=input('what is the 1-word common name for the problem you are working on? (e.g. gender for male/female classification) \n')

###############################################################
##      	      UPGRADE MODULES / LOAD MODULES             ##
###############################################################

print('-----------------------------------')
print('          LOADING MODULES          ')
print('-----------------------------------')
# upgrade to have the proper scikit-learn version later
os.chdir(cur_dir)
os.system('python3 upgrade.py')
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve

###############################################################
##                    CLEAN THE DATA                        ##
###############################################################
clean_data=settings['clean_data']
clean_dir=prevdir+'/datasets/cleaning'

if clean_data == True and mtype == 'c':
	
	# only pursue augmentation strategies on directories of files and classification problems
	print('-----------------------------------')
	print(f.renderText('CLEANING DATA'))
	print('-----------------------------------')
	for i in range(len(classes)):
		os.chdir(clean_dir)
		os.system('python3 clean.py %s %s'%(clean_dir, data_dir+'/'+classes[i]))

###############################################################
##                    AUGMENT THE DATA                       ##
###############################################################
augment_data=settings['augment_data']
augment_dir=prevdir+'/datasets/augmentation'

if augment_data == True and mtype == 'c':

	# only pursue augmentation strategies on directories of files and classification problems
	print('-----------------------------------')
	print(f.renderText('AUGMENTING DATA'))
	print('-----------------------------------')
	for i in range(len(classes)):
		os.chdir(augment_dir)
		os.system('python3 augment.py %s'%(data_dir+'/'+classes[i]))

###############################################################
##                    FEATURIZE FILES                        ##
###############################################################

# now featurize each class (in proper folder)
if mtype == 'c':
	data={}
	print('-----------------------------------')
	print(f.renderText('FEATURIZING DATA'))
	print('-----------------------------------')
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
		elif problemtype == 'csv':
			# featurize .CSV 
			os.chdir(prevdir+'/features/csv_features')
			default_features=default_csv_features

		print('-----------------------------------')
		print('           FEATURIZING %s'%(classes[i].upper()))
		print('-----------------------------------')
		
		os.system('python3 featurize.py %s'%(data_dir+'/'+classes[i]))
		os.chdir(data_dir+'/'+classes[i])
		# load audio features 
		listdir=os.listdir()
		feature_list=list()
		label_list=list()
		for j in range(len(listdir)):
			if listdir[j][-5:]=='.json':
				try:
					g=json.load(open(listdir[j]))
					# consolidate all features into one array (if featurizing with multiple featurizers)
					default_feature=list()
					default_label=list()
					for k in range(len(default_features)):
						default_feature=default_feature+g['features'][problemtype][default_features[k]]['features']
						default_label=default_label+g['features'][problemtype][default_features[k]]['labels']

					feature_list.append(default_feature)
					label_list.append(default_label)
				except:
					print('ERROR - skipping ' + listdir[j])
		
		data[class_type]=feature_list
elif mtype == 'r':
	# featurize .CSV 
	os.chdir(prevdir+'/features/csv_features')
	output_file=str(uuid.uuid1())+'.csv'
	os.system('python3 featurize_csv_regression.py --input %s --output %s'%(prevdir+'/train_dir/'+csvfile, prevdir+'/train_dir/'+output_file))
	csvfile=output_file
	default_features=['csv_regression']

###############################################################
##                  GENERATE TRAINING DATA                   ##
###############################################################

# perform class balance such that both classes have the same number
# of members (true by default, but can also be false)
os.chdir(prevdir+'/training/')
model_dir=prevdir+'/models'
balance=settings['balance_data']

if mtype == 'c':
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

		# only balance if specified in settings
		if balance==True:
			if len(class_) > minlength:
				print('%s greater than minlength (%s) by %s, equalizing...'%(classes[i], str(minlength), str(len(class_)-minlength)))
				class_=class_[0:minlength]

		for j in range(len(class_)):
			alldata.append(class_[j])
			labels.append(i)

	# load features file and get feature labels by loading in classes
	labels_dir=prevdir+'/train_dir/'+classes[0]
	os.chdir(labels_dir)
	listdir=os.listdir()
	features_file=''
	for i in range(len(listdir)):
		if listdir[i].endswith('.json'):
			features_file=listdir[i]

	labels_=list()

	for i in range(len(default_features)):
		tlabel=json.load(open(features_file))['features'][problemtype][default_features[i]]['labels']
		labels_=labels_+tlabel

elif mtype == 'r':
	regression_data=pd.read_csv(prevdir+'/train_dir/'+csvfile)
	print(csvfile)
	# get features and labels
	features_=regression_data.drop(columns=classes, axis=1)
	labels_=list(features_)
	labels_csv=regression_data.drop(columns=list(features_), axis=1)
	# iterate through each column and make into proper features and labels
	features=list()
	labels=list()

	# testing
	# print(len(features_))
	# print(len(labels_))
	# print(features_)
	# print(labels_)
	# print(features_.iloc[0,:])
	# print(labels_.iloc[0,:])

	# get features and labels 
	for i in range(len(features_)):
		features.append(list(features_.iloc[i,:]))
		labels.append(list(labels_csv.iloc[i,:]))

	# convert to name alldata just to be consistent
	alldata=features
	# print(alldata[0])
	# print(labels[0])

# print(labels_)
os.chdir(model_dir)

# get the split from the settings.json
try:
	test_size=settings['test_size'][0]
except:
	test_size=0.25

# split the data 
X_train, X_test, y_train, y_test = train_test_split(alldata, labels, test_size=test_size)

# convert everything to numpy arrays (for testing later)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

# create list of created csv files
created_csv_files=list()

# create training and testing datasets and save to a .CSV file for archive purposes
# this ensures that all machine learning training methods use the same training data
basefile=common_name
temp_listdir=os.listdir()

if create_csv == True:
	try:
		print(basefile+'_all.csv'.upper())
		if basefile+'_all.csv' not in temp_listdir:
			all_data = convert_csv(alldata, labels, labels_, mtype, classes)
			all_data.to_csv(basefile+'_all.csv',index=False)
		created_csv_files.append(basefile+'_all.csv')
	except:
		print('error exporting data into excel sheet %s'%(basefile+'_all.csv'))
	try:
		print(basefile+'_train.csv'.upper())
		if basefile+'_train.csv' not in temp_listdir:
			train_data= convert_csv(X_train, y_train, labels_, mtype, classes)
			train_data.to_csv(basefile+'_train.csv',index=False)
		created_csv_files.append(basefile+'_train.csv')
	except:
		print('error exporting data into excel sheet %s'%(basefile+'_train.csv'))
	try:
		print(basefile+'_test.csv'.upper())
		if basefile+'_test.csv' not in temp_listdir:
			test_data= convert_csv(X_test, y_test, labels_, mtype, classes)
			test_data.to_csv(basefile+'_test.csv',index=False)
		created_csv_files.append(basefile+'_test.csv')
	except:
		print('error exporting data into excel sheet %s'%(basefile+'_test.csv'))

############################################################
## 			        DATA TRANSFORMATION 			      ##
############################################################

'''
Scale features via scalers, dimensionality reduction techniques,
and feature selection strategies per the settings.json document.
'''
preprocess_dir=prevdir+'/preprocessing'
os.chdir(preprocess_dir)

# get all the important settings for the transformations 
scale_features=settings['scale_features']
reduce_dimensions=settings['reduce_dimensions']
select_features=settings['select_features']
default_scalers=settings['default_scaler']
default_reducers=settings['default_dimensionality_reducer']
default_selectors=settings['default_feature_selector']

# get command for terminal
transform_command=''
for i in range(len(classes)):
	transform_command=transform_command+' '+classes[i]

# get filename / create a unique file name
if mtype=='r':
	t_filename='r_'+common_name
elif mtype=='c':
	t_filename='c_'+common_name

# only add names in if True 
if scale_features == True:
	for i in range(len(default_scalers)):
		t_filename=t_filename+'_'+default_scalers[i]
if reduce_dimensions == True:
	for i in range(len(default_reducers)):
		t_filename=t_filename+'_'+default_reducers[i]
if select_features == True:
	for i in range(len(default_selectors)):
		t_filename=t_filename+'_'+default_selectors[i]

transform_file=t_filename+'.pickle'

if scale_features == True or reduce_dimensions == True or select_features == True:
	print('----------------------------------')
	print(f.renderText('TRANSFORMING DATA'))
	print('----------------------------------')
	# go to proper transformer directory
	try:
		os.chdir(problemtype+'_transformer')
	except:
		os.mkdir(problemtype+'_transformer')
		os.chdir(problemtype+'_transformer')
	# train transformer if it doesn't already exist
	if transform_file in os.listdir():
		os.system('pip3 install scikit-learn==0.22.2.post1')
		transform_model=pickle.load(open(transform_file,'rb'))
		alldata=transform_model.transform(alldata)
	else:
		print('making transformer...')
		os.chdir(preprocess_dir)
		if mtype == 'c':
			print('python3 transform.py %s %s %s %s'%(problemtype, 'c', common_name, transform_command))
			os.system('python3 transform.py %s %s %s %s'%(problemtype, 'c', common_name, transform_command))
			os.chdir(problemtype+'_transformer')
			transform_model=pickle.load(open(transform_file,'rb'))
			alldata=transform_model.transform(alldata)
		elif mtype == 'r':
			command='python3 transform.py %s %s %s %s %s %s'%('csv', 'r', classes[0], csvfile, prevdir+'/train_dir/', common_name)
			print(command)
			os.system(command)
			os.chdir(problemtype+'_transformer')
			transform_model=pickle.load(open(transform_file,'rb'))
			alldata=transform_model.transform(alldata)


	os.chdir(preprocess_dir)
	os.system('python3 load_transformer.py %s %s'%(problemtype, transform_file))

	# now make new files as .CSV
	os.chdir(model_dir)
	alldata=np.asarray(alldata)
	labels=np.asarray(labels)

	# split the data 
	X_train, X_test, y_train, y_test = train_test_split(alldata, labels, test_size=test_size)

	# convert to numpy arrays 
	X_train=np.array(X_train)
	X_test=np.array(X_test)
	y_train=np.array(y_train)
	y_test=np.array(y_test)
	
	# get new labels_ array 
	labels_=list()
	for i in range(len(alldata[0].tolist())):
		labels_.append('transformed_feature_%s'%(str(i)))

	# now create transformed excel sheets
	temp_listdir=os.listdir()
	if create_csv == True:
		try:
			print(basefile+'_all_transformed.csv'.upper())
			if basefile+'_all_transformed.csv' not in temp_listdir:
				all_data = convert_csv(alldata, labels, labels_, mtype, classes)
				all_data.to_csv(basefile+'_all_transformed.csv',index=False)
			created_csv_files.append(basefile+'_all_transformed.csv')
		except:
			print('error exporting data into excel sheet %s'%(basefile+'_all_transformed.csv'))
		try:
			print(basefile+'_train_transformed.csv'.upper())
			if basefile+'_train_transformed.csv' not in temp_listdir:
				train_data= convert_csv(X_train, y_train, labels_, mtype, classes)
				train_data.to_csv(basefile+'_train_transformed.csv',index=False)
			created_csv_files.append(basefile+'_train_transformed.csv')
		except:
			print('error exporting data into excel sheet %s'%(basefile+'_train_transformed.csv'))

		try:
			print(basefile+'_test_transformed.csv'.upper())
			if basefile+'_test_transformed.csv' not in temp_listdir:
				test_data= convert_csv(X_test, y_test, labels_, mtype, classes)
				test_data.to_csv(basefile+'_test_transformed.csv',index=False)
			created_csv_files.append(basefile+'_test_transformed.csv')
		except:
			print('error exporting data into excel sheet %s'%(basefile+'_test_transformed.csv'))
else:
	# make a transform model == '' so that later during model training this can be skipped
	transform_model=''

############################################################
## 					VISUALIZE DATA  					  ##
############################################################

visualize_data=settings['visualize_data']
visual_dir=prevdir+'/visualize'
model_session=str(uuid.uuid4())
os.chdir(visual_dir)

if visualize_data == True and mtype == 'c':
	print('----------------------------------')
	print(f.renderText('VISUALIZING DATA'))
	print('----------------------------------')

	command='python3 visualize.py %s'%(problemtype)
	for i in range(len(classes)):
		command=command+' '+classes[i]
	os.system(command)

	# restructure the visualization directory 
	os.chdir(visual_dir+'/visualization_session')
	os.mkdir('visualizations')
	vizdir=os.getcwd()

	# move directories so that visualization is separate from main model directory 
	shutil.move(vizdir+'/clustering', vizdir+'/visualizations/clustering')
	shutil.move(vizdir+'/feature_ranking', vizdir+'/visualizations/feature_ranking')
	shutil.move(vizdir+'/model_selection', vizdir+'/visualizations/model_selection')
	
	# go back to main direcotry
	os.chdir(visual_dir)

	# now copy over the visualization directory to 
	try:
		shutil.copytree(visual_dir+'/visualization_session', model_dir+'/'+model_session)
	except:
		shutil.rmtree(model_dir+'/'+model_session)
		shutil.copytree(visual_dir+'/visualization_session', model_dir+'/'+model_session)
	# copy over settings.json 
	shutil.copy(prevdir+'/settings.json',model_dir+'/%s/settings.json'%(model_session))

else:
	# make a model session for next section if it doesn't exist from visualization directory
	os.chdir(model_dir)
	try:
		os.mkdir(model_session)
	except:
		shutil.rmtree(model_session)
		os.mkdir(model_session)
	# copy over settings.json
	shutil.copy(prevdir+'/settings.json', model_dir+'/%s/settings.json'%(model_session))

############################################################
## 					TRAIN THE MODEL 					  ##
############################################################

'''
Now we can train the machine learning model via the default_training script.
Note you can specify multiple training scripts and it will consecutively model the 
files appropriately. 

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#  Here is what all the variables below mean:
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

# alldata = list of features in an array for model training  
	# [[39.0, 112.15384615384616, 70.98195453650514, 248.0, 14.0, 103.0, 143.5546875...],
		...
	   [39.0, 112.15384615384616, 70.98195453650514, 248.0, 14.0, 103.0, 143.5546875,...]]
# labels = list of labels in an array for model training 
	# ['males','females',...,'males','females']
# mtype = classification or regression problem?
	# 'c' --> classification
	# 'r' --> regression
# jsonfile = filename of the .JSON document seprating classes
	#  males_females.json
# problemtype = type of problem selected
	# 'audio' --> audio files
	# 'image' --> images files
	# 'text' --> text files
	# 'video' --> video files
	# 'csv' --> csv files 
# default_featurenames = default feature array(s) to use for modeling 
	# ['librosa_features']
# settings = overall settings currenty used for model training 
	# output of the settings.json document 

-----

# transform_model = transformer model if applicable 
	# useful for data transformation as part of the model initialization process (if pickle file)
	# uses scikit-learn pipeline 

# X_train, X_test, y_train, y_test
	# training datasets used in the .CSV documents 
	# also can use pandas dataframe if applicable (loading in the model dir)
'''
print('----------------------------------')
print(f.renderText('MODELING DATA'))
print('----------------------------------')

# get defaults 
default_training_scripts=settings['default_training_script']
model_compress=settings['model_compress']
default_featurenames=''

for i in range(len(default_features)):
	if i ==0:
		default_featurenames=default_features[i]
	else:
		default_featurenames=default_featurenames+'_|_'+default_features[i] 

# just move all created .csv files into model_session directory
os.chdir(model_dir)
os.chdir(model_session)
os.mkdir('data')
for i in range(len(created_csv_files)):
	shutil.move(model_dir+'/'+created_csv_files[i], os.getcwd()+'/data/'+created_csv_files[i])

# initialize i (for tqdm) and go through all model training scripts 
i=0
for i in tqdm(range(len(default_training_scripts)), desc=default_training_scripts[i]):

	try:
		# go to model directory 
		os.chdir(model_dir)

		# get common name and default training script to select proper model trainer
		default_training_script=default_training_scripts[i]
		common_name_model=common_name+'_'+default_training_script

		print('----------------------------------')
		print('       .... training %s           '%(default_training_script.upper()))
		print('----------------------------------')

		if default_training_script=='adanet':
			print('Adanet training is coming soon! Please use a different model setting for now.') 
			# import train_adanet as ta 
			# ta.train_adanet(mtype, classes, jsonfile, alldata, labels, feature_labels, problemtype, default_featurenames)
		elif default_training_script=='alphapy':
			import train_alphapy as talpy
			modelname, modeldir, files=talpy.train_alphapy(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='atm':
			import train_atm as tatm
			modelname, modeldir, files=tatm.train_atm(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='autobazaar':
			import train_autobazaar as autobzr
			modelname, modeldir, files=autobzr.train_autobazaar(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='autogbt':
			import train_autogbt as tautogbt
			modelname, modeldir, files=tautogbt.train_autogbt(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='autogluon':
			import train_autogluon as tautg
			modelname, modeldir, files, test_data=tautg.train_autogluon(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='autokaggle':
			import train_autokaggle as autokag
			modelname, modeldir, files=autokag.train_autokaggle(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='autokeras':
			import train_autokeras as autokeras_
			modelname, modeldir, files=autokeras_.train_autokeras(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='automl':
			import train_automl as auto_ml
			modelname, modeldir, files=auto_ml.train_automl(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='autosklearn':
			print('Autosklearn training is unstable! Please use a different model setting for now.') 
			# import train_autosklearn as taskl
			# taskl.train_autosklearn(alldata, labels, mtype, jsonfile, problemtype, default_featurenames)
		elif default_training_script=='autopytorch':
			import train_autopytorch as autotorch_
			modelname, modeldir, files=autotorch_.train_autopytorch(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='btb':
			import train_btb as tbtb
			modelname, modeldir, files=tbtb.train_btb(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='cvopt':
			import train_cvopt as tcvopt
			modelname, modeldir, files = tcvopt.train_cvopt(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='devol':
			import train_devol as td 
			modelname, modeldir, files=td.train_devol(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='gama':
			import train_gama as tgama
			modelname, modeldir, files=tgama.train_gama(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='gentun':
			import train_gentun as tgentun 
			modelname, modeldir, files=tgentun.train_gentun(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='hyperband':
			import train_hyperband as thband
			modelname, modeldir, files = thband.train_hyperband(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='hypsklearn':
			import train_hypsklearn as th 
			modelname, modeldir, files=th.train_hypsklearn(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='hungabunga':
			import train_hungabunga as thung
			modelname, modeldir, files=thung.train_hungabunga(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='imbalance':
			import train_imbalance as timb
			modelname, modeldir, files=timb.train_imbalance(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='keras':
			import train_keras as tk
			modelname, modeldir, files=tk.train_keras(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='ludwig':
			import train_ludwig as tl
			modelname, modeldir, files=tl.train_ludwig(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='mlblocks':
			import train_mlblocks as mlb
			modelname, modeldir, files=mlb.train_mlblocks(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='mlbox':
			import train_mlbox as mlbox_
			modelname, modeldir, files=mlbox_.train_mlbox(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='neuraxle':
			if mtype=='c':
				print('Neuraxle does not support classification at this time. Please use a different model training script')
				break
			else:
				import train_neuraxle as tneuraxle
				modelname, modeldir, files=tneuraxle.train_neuraxle(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='plda':
			print('PLDA training is unstable! Please use a different model setting for now.') 
			# import train_pLDA as tp
			# tp.train_pLDA(alldata,labels)
		elif default_training_script=='pytorch':
			import train_pytorch as t_pytorch
			modelname, modeldir, files = t_pytorch.train_pytorch(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='safe':
			import train_safe as tsafe
			modelname, modeldir, files=tsafe.train_safe(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)
		elif default_training_script=='scsr':
			import train_scsr as scsr
			if mtype == 'c':
				modelname, modeldir, files=scsr.train_sc(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,minlength)
			elif mtype == 'r':
				modelname, modeldir, files=scsr.train_sr(X_train,X_test,y_train,y_test,common_name_model,problemtype,classes,default_featurenames,transform_model,model_dir,settings)
		elif default_training_script=='tpot':
			import train_TPOT as tt
			modelname, modeldir, files=tt.train_TPOT(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session)

		############################################################
		## 		  CALCULATE METRICS / PLOT ROC CURVE        	  ##
		############################################################

		if modelname.endswith('.pickle'):
			foldername=modelname[0:-7]
		elif modelname.endswith('.h5'):
			foldername=modelname[0:-3]
		else:
			foldername=common_name_model

		# copy the folder in case there are multiple models being trained 
		try:
			shutil.copytree(model_session, foldername)
		except:
			shutil.rmtree(foldername)
			shutil.copytree(model_session, foldername)
			
		cur_dir2=os.getcwd()
		os.chdir(foldername)
		os.mkdir('model')
		os.chdir('model')
		model_dir_temp=os.getcwd()

		# dump transform model to the models directory if necessary
		if transform_model == '':
			transformer_name=''
		else:
			# dump the tranform model into the current working directory
			transformer_name=modelname.split('.')[0]+'_transform.pickle'
			tmodel=open(transformer_name,'wb')
			pickle.dump(transform_model, tmodel)
			tmodel.close()

		# move all supplementary files into model folder
		for j in range(len(files)):
			shutil.move(modeldir+'/'+files[j], model_dir_temp+'/'+files[j])
		
		# load model for getting metrics
		if default_training_script not in ['alphapy', 'atm', 'autokeras', 'autopytorch', 'ludwig', 'keras', 'devol']:
			loadmodel=open(modelname, 'rb')
			clf=pickle.load(loadmodel)
			loadmodel.close()
		elif default_training_script == 'atm':
			from atm import Model
			clf=Model.load(modelname)
		elif default_training_script == 'autokeras':
			import tensorflow as tf
			import autokeras as ak
			clf = pickle.load(open(modelname, 'rb'))
		elif default_training_script=='autopytorch':
			import torch
			clf=torch.load(modelname)
		elif default_training_script == 'ludwig':
			from ludwig.api import LudwigModel
			clf=LudwigModel.load('ludwig_files/experiment_run/model/')
		elif default_training_script in ['devol', 'keras']: 
			from keras.models import load_model
			clf = load_model(modelname)
		else: 
			clf=''

		# create test_data variable for anything other than autogluon
		if default_training_script != 'autogluon':
			test_data=''

		# now make main .JSON file for the session summary with metrics
		get_metrics(clf, problemtype, mtype, default_training_script, common_name, X_test, y_test, classes, modelname, settings, model_session, transformer_name, created_csv_files, test_data)
		
		# now move to the proper models directory
		os.chdir(model_dir)

		try:
			os.chdir(problemtype+'_models')
		except:
			os.mkdir(problemtype+'_models')
			os.chdir(problemtype+'_models')

		shutil.move(model_dir+'/'+foldername, os.getcwd()+'/'+foldername)
		
		############################################################
		## 					COMPRESS MODELS 					  ##
		############################################################

		if model_compress == True:
			print(f.renderText('COMPRESSING MODEL'))
			# now compress the model according to model type 
			if default_training_script in ['hypsklearn', 'scsr', 'tpot']:
				# all .pickle files and can compress via scikit-small-ensemble
				from sklearn.externals import joblib

				# open up model 
				loadmodel=open(modelname, 'rb')
				model = pickle.load(loadmodel)
				loadmodel.close()

				# compress - from 0 to 9. Higher value means more compression, but also slower read and write times. 
				# Using a value of 3 is often a good compromise.
				joblib.dump(model, modelname[0:-7]+'_compressed.joblib',compress=3)

				# can now load compressed models as such
				# thenewmodel=joblib.load(modelname[0:-7]+'_compressed.joblib')
				# leads to up to 10x reduction in model size and .72 sec - 0.23 secoon (3-4x faster loading model)
				# note may note work in sklearn and python versions are different from saving and loading environments. 

			elif default_training_script in ['devol', 'keras']: 
				# can compress with keras_compressor 
				import logging
				from keras.models import load_model
				from keras_compressor.compressor import compress

				logging.basicConfig(
					level=logging.INFO,
				)

				try:
					print('compressing model!!')
					model = load_model(modelname)
					model = compress(model, 7e-1)
					model.save(modelname[0:-3]+'_compressed.h5')
				except:
					print('error compressing model!!')

			else:
				# for everything else, we can compress pocketflow models in the future.
				print('We cannot currently compress %s models. We are working on this!! \n\n The model will remain uncompressed for now'%(default_training_script))


		############################################################
		## 					CREATE YAML FILES 					  ##
		############################################################

		create_YAML=settings['create_YAML']

		# note this will only work for pickle files for now. Will upgrade to deep learning
		# models in the future.
		if create_YAML == True and default_training_script in ['scsr', 'tpot', 'hypsklearn'] and problemtype=='audio':
			print(f.renderText('PRODUCTIONIZING MODEL'))
			# clear the load directory 
			os.chdir(prevdir+'/load_dir/')
			listdir=os.listdir()
			for i in range(len(listdir)):
				try:
					# for any files in the ./load_dir 
					os.remove(listdir[i])
				except:
					# for folders in the ./load_dir
					try:
						shutil.rmtree(listdir[i])
					except:
						print('error, %s not a folder'%(listdir[i]))

			# First need to create some test files for each use case
			# copy 10 files from each class into the load_dir 
			copylist=list()
			print(classes)

			# clear the load_directory 
			os.chdir(prevdir+'/load_dir/')
			listdir2=os.listdir()
			for j in range(len(listdir2)):
				# remove all files in the load_dir 
				os.remove(listdir2[j])	

			for i in range(len(classes)):
				os.chdir(prevdir+'/train_dir/'+classes[i])
				tempdirectory=os.getcwd()
				listdir=os.listdir()
				os.chdir(tempdirectory)
				count=0
				
				# rename files if necessary 
				# the only other file in the directory should be .json, so this should work
				for j in range(len(listdir)):
					# print(listdir[j])
					if listdir[j].replace(' ','_').replace('-','') not in copylist and listdir[j].endswith('.json') == False:
						if count >= 10:
							break
						else:
							newname=listdir[j].replace(' ','_').replace('-','')
							os.rename(listdir[j], newname)
							shutil.copy(tempdirectory+'/'+newname, prevdir+'/load_dir/'+newname)
							# print(tempdirectory+'/'+listdir[j])
							copylist.append(newname)
							count=count+1 

					elif listdir[j].replace(' ','_').replace('-','') in copylist and listdir[j].endswith('.json') == False: 
						# rename the file appropriately such that there are no conflicts.
						if count >= 10:
							break
						else:
							newname=str(j)+'_'+listdir[j].replace(' ','_').replace('-','')
							os.rename(listdir[j], newname)
							# print(tempdirectory+'/'+newname)
							shutil.copy(tempdirectory+'/'+newname, prevdir+'/load_dir/'+newname)
							copylist.append(newname)
							count=count+1 

			# now apply machine learning model to these files 
			os.chdir(prevdir+'/models')
			os.system('python3 load.py')

			# now load up each of these and break loop when tests for each class exist 
			os.chdir(prevdir+'/load_dir')
			jsonfiles=list()
			# iterate through until you find a class that fits in models 
			listdir=os.listdir()

			for i in range(len(classes)):
				for j in range(len(listdir)):
					if listdir[j][-5:]=='.json':
						try:
							g=json.load(open(listdir[j]))
							models=g['models'][problemtype][classes[i]]
							print(models)
							features=list()
							for k in range(len(default_features)):
								features=features+g['features'][problemtype][default_features[k]]['features']
							
							print(features)
							testfile=classes[i]+'_features.json'
							print(testfile)
							jsonfile2=open(testfile,'w')
							json.dump(features,jsonfile2)
							jsonfile2.close()
							shutil.move(os.getcwd()+'/'+testfile, prevdir+'/models/'+problemtype+'_models/'+testfile)
							break
						except:
							pass 

			# now create the YAML file 
			os.chdir(prevdir+'/production')
			cmdclasses=''
			for i in range(len(classes)):
				if i != len(classes)-1:
					cmdclasses=cmdclasses+classes[i]+' '
				else:
					cmdclasses=cmdclasses+classes[i]

			if 'common_name' in locals():
				pass 
			else:
				common_name=modelname[0:-7]

				jsonfilename=modelname[0:-7]+'.json'
				os.system('python3 create_yaml.py %s %s %s %s %s'%(problemtype, common_name, modelname, jsonfilename, cmdclasses))
	except:
		print('ERROR - error in modeling session')