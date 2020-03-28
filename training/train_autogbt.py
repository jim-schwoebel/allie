from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random, pickle, time, json
import numpy as np
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from autogbt import AutoGBTClassifier

def train_autogbt(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):

	print('creating training data...')
	x_train, x_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)

	# mae a binarizer from set of labels 
	# make labels into some binary form 
	lb = LabelBinarizer()
	lb.fit(list(range(len(set(labels)))))

	# create binarizer
	y_train = lb.transform(y_train)
	y_test = lb.transform(y_test)

	# scale data to allow for convergence to happen faster for SVM
	scaler = StandardScaler()
	scaler.fit(x_train)

	# create scaler 
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	# classiifcation
	model = AutoGBTClassifier()
	model.fit(x_train, y_train)

	roc_score=roc_auc_score(y_test, model.predict(x_test))
	model_score=model.best_score

	print('valid AUC: %.3f' % (roc_score))
	print('CV AUC: %.3f' % (model_score))

	modelname=jsonfile[0:-5]+'_autogbt_%s'%(str(default_features.replace("'",'').replace('"','')))

	# save the models 
	scaler_modelname=modelname+'scaler.pickle'
	binarizer_modelname=modelname+'binarizer.pickle'
	main_modelname=modelname+'model.pickle'

	print('saving scaler...')
	smodel=open(scaler_modelname,'wb')
	g=pickle.dump(scaler, smodel)
	smodel.close() 

	print('saving binarizer...')
	bmodel=open(binarizer_modelname,'wb')
	g=pickle.dump(lb, bmodel)
	bmodel.close() 

	print('saving model...')
	pmodel=open(main_modelname,'wb')
	g=pickle.dump(model, pmodel)
	pmodel.close()

	# JSONFILE
	jsonfile=open(modelname+'.json','w')
	data={'roc_score': roc_score,
		  'model_score': model_score,
		  'sampletype': problemtype,
		  'feature_set': default_features,
		  'model_name': main_modelname,
		  'binarizer_modelname': binarizer_modelname,
		  'scaler_modelname': scaler_modelname,
		  'training_type': 'autogbt',
		  'settings': settings,
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
	shutil.move(cur_dir2+'/'+scaler_modelname, os.getcwd()+'/'+scaler_modelname)
	shutil.move(cur_dir2+'/'+binarizer_modelname, os.getcwd()+'/'+binarizer_modelname)
	shutil.move(cur_dir2+'/'+main_modelname, os.getcwd()+'/'+main_modelname)

	# making predictions 
	# print(model.model.predict(x_train[0].reshape(1,28,28,1)))
	# print(y_train[0])