'''
               AAA               lllllll lllllll   iiii                      
              A:::A              l:::::l l:::::l  i::::i                     
             A:::::A             l:::::l l:::::l   iiii                      
            A:::::::A            l:::::l l:::::l                             
           A:::::::::A            l::::l  l::::l iiiiiii     eeeeeeeeeeee    
          A:::::A:::::A           l::::l  l::::l i:::::i   ee::::::::::::ee  
         A:::::A A:::::A          l::::l  l::::l  i::::i  e::::::eeeee:::::ee
        A:::::A   A:::::A         l::::l  l::::l  i::::i e::::::e     e:::::e
       A:::::A     A:::::A        l::::l  l::::l  i::::i e:::::::eeeee::::::e
      A:::::AAAAAAAAA:::::A       l::::l  l::::l  i::::i e:::::::::::::::::e 
     A:::::::::::::::::::::A      l::::l  l::::l  i::::i e::::::eeeeeeeeeee  
    A:::::AAAAAAAAAAAAA:::::A     l::::l  l::::l  i::::i e:::::::e           
   A:::::A             A:::::A   l::::::ll::::::li::::::ie::::::::e          
  A:::::A               A:::::A  l::::::ll::::::li::::::i e::::::::eeeeeeee  
 A:::::A                 A:::::A l::::::ll::::::li::::::i  ee:::::::::::::e  
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee  
                                                                             
|  \/  |         | |    | |  / _ \ | ___ \_   _|
| .  . | ___   __| | ___| | / /_\ \| |_/ / | |  
| |\/| |/ _ \ / _` |/ _ \ | |  _  ||  __/  | |  
| |  | | (_) | (_| |  __/ | | | | || |    _| |_ 
\_|  |_/\___/ \__,_|\___|_| \_| |_/\_|    \___/ 

Train models using BTB: https://github.com/HDI-Project/BTB

This is enabled if the default_training_script = ['btb']
'''
import warnings, datetime, uuid, os, json, shutil, pickle

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pandas as pd

os.system('pip3 install baytune==0.3.7')
# os.system('pip3 install autobazaar==0.2.0')
# os.system('pip3 install gitpython==3.0.2')
# os.system('pip3 install --upgrade GitPython==2.1.15')
# os.system('pip3 install --upgrade gitdb2==2.0.6 gitdb==0.6.4 ')

# make imports 
print('installing package configuration')
from btb.session import BTBSession
from btb.tuning import Tunable
from btb.tuning.tuners import GPTuner
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from btb.selection import UCB1
from btb.tuning.hyperparams import FloatHyperParam, IntHyperParam

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

def train_btb(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

	# create file names
	model_name=common_name_model+'.pickle'
	folder='btb_session'
	csvname=common_name_model.split('_')[0]
	curdir=os.getcwd()
	files=list()

	# make a temporary folder for the training session
	try:
		os.mkdir(folder)
		os.chdir(folder)
	except:
		shutil.rmtree(folder)
		os.mkdir(folder)
		os.chdir(folder)

	# get training and testing data
	try:
		shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_train_transformed.csv',os.getcwd()+'/train.csv')
		shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_test_transformed.csv',os.getcwd()+'/test.csv')
	except:
		shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_train.csv',os.getcwd()+'/train.csv')  
		shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_test.csv',os.getcwd()+'/test.csv')

	# create required .JSON
	dataset_id, filename=create_json(folder, 'train.csv')
	os.mkdir(dataset_id)
	os.chdir(dataset_id)
	os.mkdir('tables')
	shutil.copy(curdir+'/'+folder+'/train.csv', os.getcwd()+'/tables/train.csv')

	if mtype=='c':

		def build_model(name, hyperparameters):
			model_class = models[name]
			return model_class(random_state=0, **hyperparameters)

		def score_model(name, hyperparameters):
			model = build_model(name, hyperparameters)
			scores = cross_val_score(model, X_train, y_train)
			return scores.mean()

		rf_hyperparams = {'n_estimators': IntHyperParam(min=10, max=500),
						'max_depth': IntHyperParam(min=10, max=500)}

		rf_tunable = Tunable(rf_hyperparams)
		print(rf_tunable)

		svc_hyperparams = {'C': FloatHyperParam(min=0.01, max=10.0),
							'gamma': FloatHyperParam(0.000000001, 0.0000001)}

		svc_tunable = Tunable(svc_hyperparams)
		print(svc_tunable)

		tuners = {'RF': rf_tunable,
				  'SVC': svc_tunable}

		print(tuners)

		models = {'RF': RandomForestClassifier,
				  'SVC': SVC}

		selector = UCB1(['RF', 'SVC'])

		session = BTBSession(tuners, score_model, verbose=True)
		best_proposal = session.run(iterations=100)  
		best_model = build_model(best_proposal['name'], best_proposal['config'])
		best_model.fit(X_train, y_train)
		accuracy =  best_model.score(X_test, y_test)

		# tuner.record(parameters, score)
		print('ACCURACY:')
		print(accuracy)

		# now save the model in .pickle
		os.chdir(curdir)
		f=open(model_name,'wb')
		pickle.dump(best_model, f)
		f.close()


	elif mtype == 'r':


		tunables = {
			'random_forest': {
				'n_estimators': {'type': 'int', 'default': 2, 'range': [1, 1000]},
				'max_features': {'type': 'str', 'default': 'log2', 'range': [None, 'auto', 'log2', 'sqrt']},
				'min_samples_split': {'type': 'int', 'default': 2, 'range': [2, 20]},
				'min_samples_leaf': {'type': 'int', 'default': 2, 'range': [1, 20]},
			},
			'extra_trees': {
				'n_estimators': {'type': 'int', 'default': 2, 'range': [1, 1000]},
				'max_features': {'type': 'str', 'default': 'log2', 'range': [None, 'auto', 'log2', 'sqrt']},
				'min_samples_split': {'type': 'int', 'default': 2, 'range': [2, 20]},
				'min_samples_leaf': {'type': 'int', 'default': 2, 'range': [1, 20]},
			}
		}

		models = {
			'random_forest': RandomForestRegressor,
			'extra_trees': ExtraTreesRegressor,
		}

		def build_model(name, hyperparameters):
			model_class = models[name]
			return model_class(random_state=0, **hyperparameters)

		def score_model(name, hyperparameters):
			model = build_model(name, hyperparameters)
			r2_scorer = make_scorer(r2_score)
			scores = cross_val_score(model, X_train, y_train, scoring=r2_scorer)
			return scores.mean()


		session = BTBSession(tunables, score_model, verbose=True)
		best_proposal = session.run(iterations=100)  
		best_model = build_model(best_proposal['name'], best_proposal['config'])

		best_model.fit(X_train, y_train)
		pred = best_model.predict(X_test)

		r2_score=r2_score(y_test, pred)

		print('R2 score!!')
		print(r2_score)
		
		# now save the model in .pickle
		os.chdir(curdir)
		f=open(model_name,'wb')
		pickle.dump(best_model, f)
		f.close()

	files.append(model_name)
	files.append(folder)
	model_dir=os.getcwd()

	return model_name, model_dir, files
