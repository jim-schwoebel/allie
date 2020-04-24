import warnings, datetime, uuid, os, json, shutil, pickle

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pandas as pd


'''
Taken from the example here:
https://github.com/HDI-Project/BTB/blob/master/notebooks/BTBSession%20-%20Example.ipynb

Note that autobazaar is used as the primary model trainer for BTB sessions.
https://github.com/HDI-Project/AutoBazaar

Tutorial:
https://hdi-project.github.io/AutoBazaar/readme.html#install

Data: Must be formatted (https://github.com/mitll/d3m-schema/blob/master/documentation/datasetSchema.md)

Case 1: Single table
In many openml and other tabular cases, all the learning data is contained in a single tabular file. In this case, an example dataset will look like the following.

<dataset_id>/
|-- tables/
	|-- learningData.csv
		d3mIndex,sepalLength,sepalWidth,petalLength,petalWidth,species
		0,5.2,3.5,1.4,0.2,I.setosa
		1,4.9,3.0,1.4,0.2,I.setosa
		2,4.7,3.2,1.3,0.2,I.setosa
		3,4.6,3.1,1.5,0.2,I.setosa
		4,5.0,3.6,1.4,0.3,I.setosa
		5,5.4,3.5,1.7,0.4,I.setosa
		...
'''

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

def convert_(X_train, y_train, labels):

	feature_list=labels
	data=dict()

	print(len(feature_list))
	print(len(X_train[0]))
	# time.sleep(50)

	for i in range(len(X_train)):
		for j in range(len(feature_list)-1):
			if i > 0:
				# print(data[feature_list[j]])
				try:
					# print(feature_list[j])
					# print(data)
					# print(X_train[i][j])
					# print(data[feature_list[j]])
					# time.sleep(2)
					data[feature_list[j]]=data[feature_list[j]]+[X_train[i][j]]
				except:
					pass
					# print(data)
					# time.sleep(50)
					# print(str(i)+'-i')
					# print(j)

			else:
				data[feature_list[j]]=[X_train[i][j]]
				print(data)

	data['class_']=y_train
	data=pd.DataFrame(data, columns = list(data))
	print(data)
	print(list(data))
	# time.sleep(500)

	return data

def create_json(foldername, trainingcsv):

	# create the template .JSON file necessary for the featurization
	dataset_name=foldername
	dataset_id=str(uuid.uuid4())
	columns=list()

	colnames=list(pd.read_csv(trainingcsv))

	for i in range(len(colnames)):
		if colnames[i] != 'class_':
			columns.append({"colIndex": i,
					    "colName": colnames[i],
					    "colType": "real",
					    "role": ["attribute"]})
		else:
			columns.append({"colIndex": i,
					    "colName": 'class_',
					    "colType": "real",
					    "role": ["suggestedTarget"]})	

	data={"about": 
	  {
	  "datasetID": dataset_id,
	  "datasetName":dataset_name,
	  "humanSubjectsResearch": False,
	  "license":"CC",
	  "datasetSchemaVersion":"3.0",
	  "redacted":False
	  },
	"dataResources":
	  [
	    {
	      "resID": "0",
	      "resPath": os.getcwd()+'/'+trainingcsv,
	      "resType": "table",
	      "resFormat": ["text/csv"],
	      "isCollection": False,
	      "columns":columns,
	    }
	  ]
	}

	filename='datasetDoc.json'
	jsonfile=open(filename,'w')
	json.dump(data,jsonfile)
	jsonfile.close()

	return dataset_id, filename

def train_autobazaar(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):
	print('installing package configuration')
	os.system('pip3 install baytune==0.3.7')
	os.system('pip3 install autobazaar==0.2.0')
	os.system('pip3 install gitpython==3.0.2')
	os.system('pip3 install --upgrade GitPython==2.1.15')
	os.system('pip3 install --upgrade gitdb2==2.0.6 gitdb==0.6.4 ')

	# make imports 
	from btb.session import BTBSession
	from btb.tuning import Tunable
	from btb.tuning.tuners import GPTuner
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import make_scorer, r2_score
	from sklearn.model_selection import cross_val_score
	from sklearn.svm import SVC
	from btb.selection import UCB1
	from btb.tuning.hyperparams import FloatHyperParam, IntHyperParam

	# get train and test data
	print('creating training data')
	X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)

	# create file names
	model_name=jsonfile[0:-5]+'_'+str(default_features).replace("'",'').replace('"','')+'_btb'

	if mtype == 'c':
		model_name=model_name+'_classification'
		mtype='classification'
	elif mtype == 'r':
		model_name=model_name+'_regression'
		mtype='regression'

	folder=model_name
	jsonfilename=model_name+'.json'
	csvfilename=model_name+'.csv'
	trainfile=model_name+'_train.csv'
	testfile=model_name+'_test.csv'
	model_name=model_name+'.pickle'

	# this should be the model directory
	hostdir=os.getcwd()

	# open a sample featurization 
	labels_dir=prev_dir(hostdir)+'/train_dir/'+jsonfilename.split('_')[0]
	os.chdir(labels_dir)
	listdir=os.listdir()
	features_file=''
	for i in range(len(listdir)):
		if listdir[i].endswith('.json'):
			features_file=listdir[i]

	# load features file and get labels 
	labels_=json.load(open(features_file))['features'][problemtype][default_features]['labels']
	os.chdir(hostdir)

	# make a temporary folder for the training session
	try:
		os.mkdir(folder)
		os.chdir(folder)
	except:
		shutil.rmtree(folder)
		os.mkdir(folder)
		os.chdir(folder)

	all_data = convert_(alldata, labels, labels_)
	train_data= convert_(X_train, y_train, labels_)
	test_data= convert_(X_test, y_test, labels_)
	all_data.to_csv(csvfilename)
	data=pd.read_csv(csvfilename)
	os.remove(csvfilename)
	train_data.to_csv(trainfile)
	test_data.to_csv(testfile)

	dataset_id, filename=create_json(folder, trainfile)

	abz_dir=os.getcwd()

	os.mkdir(dataset_id)
	os.chdir(dataset_id)
	os.mkdir('tables')
	shutil.copy(hostdir+'/'+folder+'/'+trainfile, os.getcwd()+'/tables/'+trainfile)
	
	if mtype=='classification':

		# now save the model in .pickle
		f=open(model_name,'wb')
		pickle.dump(best_model, f)
		f.close()
		
		# SAVE JSON FILE 
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'accuracy': float(accuracy),
			'model type':'BTB_%s'%(mtype),
			'settings': settings,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

	elif mtype == 'regression':
		
		# now save the model in .pickle
		f=open(model_name,'wb')
		pickle.dump(best_model, f)
		f.close()
		
		# save the .JSON file
		print('saving .JSON file (%s)'%(jsonfilename))
		jsonfile=open(jsonfilename,'w')

		data={'sample type': problemtype,
			'feature_set':default_features,
			'model name':jsonfilename[0:-5]+'.pickle',
			'r2_score': float(r2_score),
			'model type':'BTB_%s'%(mtype),
			'settings': settings,
		}

		json.dump(data,jsonfile)
		jsonfile.close()

	# tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)

	# now get all them transferred
	os.chdir(hostdir)
	try:
		os.chdir(problemtype+'_models')
	except:
		os.mkdir(problemtype+'_models')
		os.chdir(problemtype+'_models')

	# now move all the files over to proper model directory 
	shutil.copy(hostdir+'/'+folder+'/'+dataset_id+'/'+model_name, hostdir+'/%s_models/%s'%(problemtype,model_name))
	shutil.copy(hostdir+'/'+folder+'/'+dataset_id+'/'+jsonfilename, hostdir+'/%s_models/%s'%(problemtype,jsonfilename))

	# get variables
	model_dir=hostdir+'/%s_models/'%(problemtype)

	return model_name, model_dir
