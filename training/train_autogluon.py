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

Train models using autogluon: https://github.com/awslabs/autogluon

This is enabled if the default_training_script = ['autogluon']
'''
import os 
# install dependencies
os.system('pip3 install autogluon==0.0.6')
os.system('pip3 install pillow==7.0.0')
os.system('pip3 install numpy==1.18.4')
from autogluon import TabularPrediction as task
import pandas as pd
import os, sys, pickle, json, random, shutil, time
import numpy as np

def convert_gluon(X_train, y_train):

	feature_list=list()
	for i in range(len(X_train[0])):
		feature_list.append('feature_'+str(i))

	feature_list.append('class')
	data=dict()

	for i in range(len(X_train)):
		for j in range(len(feature_list)-1):
			if i > 0:
				try:
					data[feature_list[j]]=data[feature_list[j]]+[X_train[i][j]]
				except:
					pass

			else:
				data[feature_list[j]]=[X_train[i][j]]
				print(data)

	data['class']=y_train
	data=pd.DataFrame(data, columns = list(data))
	data=task.Dataset(data)

	return data

def train_autogluon(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):
	# get train and test data 
	train_data = convert_gluon(X_train, y_train)
	test_data = convert_gluon(X_test, y_test)
	predictor = task.fit(train_data=train_data, label='class')

	# get summary
	results = predictor.fit_summary(verbosity=3)

	# get model name
	files=list()
	model_name=common_name_model+'.pickle'
	# pickle store classifier
	f=open(model_name,'wb')
	pickle.dump(predictor, f)
	f.close()

	# now rename current directory with models (keep this info in a folder)
	files.append(model_name)
	files.append('AutogluonModels')
	files.append('catboost_info')
	files.append('dask-worker-space')

	# get model_name 
	model_dir=os.getcwd()

	return model_name, model_dir, files, test_data
