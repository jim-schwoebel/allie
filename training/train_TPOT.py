import os, sys, pickle, json, random, shutil, time
import numpy as np
os.system('pip3 install tpot=0.11.3')
from tpot import TPOTClassifier
from tpot import TPOTRegressor
	
def train_TPOT(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_features,transform_model,settings, model_session):
			   
	# get modelname 
	modelname=common_name_model
	files=list()
	
	if mtype in ['classification', 'c']:
		tpot=TPOTClassifier(generations=10, population_size=50, verbosity=2, n_jobs=-1, scoring='accuracy')
		tpotname='%s_classifier.py'%(modelname)
	elif mtype in ['regression','r']:
		tpot = TPOTRegressor(generations=10, population_size=20, verbosity=2)
		tpotname='%s_regression.py'%(modelname)

	# fit classifier
	tpot.fit(X_train, y_train)
	tpot.export(tpotname)

	# export data to .json format (use all data to improve model accuracy, as it's already tested)
	data={
		'data': X_train.tolist(),
		'labels': y_train.tolist(),
	}

	jsonfilename='%s.json'%(tpotname[0:-3])
	jsonfile=open(jsonfilename,'w')
	json.dump(data,jsonfile)
	jsonfile.close()

	# now edit the file and run it 
	g=open(tpotname).read()
	g=g.replace("import numpy as np", "import numpy as np \nimport json, pickle")
	g=g.replace("tpot_data = pd.read_csv(\'PATH/TO/DATA/FILE\', sep=\'COLUMN_SEPARATOR\', dtype=np.float64)","g=json.load(open('%s'))\ntpot_data=np.array(g['labels'])"%(jsonfilename))
	g=g.replace("features = tpot_data.drop('target', axis=1)","features=np.array(g['data'])\n")
	g=g.replace("tpot_data['target'].values", "tpot_data")
	g=g.replace("results = exported_pipeline.predict(testing_features)", "print('saving classifier to disk')\nf=open('%s','wb')\npickle.dump(exported_pipeline,f)\nf.close()"%(jsonfilename[0:-5]+'.pickle'))
	g1=g.find('exported_pipeline = ')
	g2=g.find('exported_pipeline.fit(training_features, training_target)')
	g=g.replace('.values','')
	g=g.replace("tpot_data['target']",'tpot_data')
	modeltype=g[g1:g2]
	os.remove(tpotname)
	t=open(tpotname,'w')
	t.write(g)
	t.close()
	print('')
	os.system('python3 %s'%(tpotname))
	# now write an accuracy label 
	os.remove(jsonfilename)

	# get model_name 
	model_dir=os.getcwd()
	model_name=tpotname[0:-3]+'.pickle'

	# tpot file will be here now 
	files.append(tpotname)
	files.append(model_name)
	
	return model_name, model_dir, files
