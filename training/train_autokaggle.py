import pickle
curdir=os.getcwd()
print(os.getcwd())
print('initializing installation')
os.system('pip3 install autokaggle==0.1.0')
from helpers.autokaggle.tabular_supervised import TabularClassifier
from helpers.autokaggle.tabular_supervised import TabularRegressor
os.chdir(curdir)

def train_autokaggle(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):
	
	model_name=common_name_model+'.pickle'
	files=list()

	if mtype in ['classification', 'c']:

		# fit classifier 
		clf = TabularClassifier()
		clf.fit(X_train, y_train, time_limit=12 * 60 * 60)

		# SAVE ML MODEL
		modelfile=open(model_name,'wb')
		pickle.dump(clf, modelfile)
		modelfile.close()

	elif mtype in ['regression', 'r']:

		print("Starting AutoKaggle")
		clf = TabularRegressor()
		clf.fit(X_train, y_train, time_limit=12 * 60 * 60)

		# saving model
		print('saving model')
		modelfile=open(model_name,'wb')
		pickle.dump(clf, modelfile)
		modelfile.close()

	model_dir=os.getcwd()
	files.append(model_name)

	return model_name, model_dir, files