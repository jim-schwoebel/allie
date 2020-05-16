from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random, pickle, time, json, os, shutil
import numpy as np
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split

try:
	from autogbt import AutoGBTClassifier
except:
	print('initializing installation...')
	os.system('pip3 install git+https://github.com/pfnet-research/autogbt-alt.git')
	from autogbt import AutoGBTClassifier

def train_autogbt(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):
	# make initial names and lists
	files=list()
	model_name=common_name_model+'.pickle'

	# train classifier
	model = AutoGBTClassifier()
	model.fit(X_train, y_train)

	print('saving model...')
	pmodel=open(model_name,'wb')
	g=pickle.dump(model, pmodel)
	pmodel.close()

	files.append(model_name)
	model_dir=os.getcwd()

	return model_name, model_dir, files