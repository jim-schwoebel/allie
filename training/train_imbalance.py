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

Train models using SMOTE/imbalance-learn: https://pypi.org/project/imbalanced-learn/

This is enabled if the default_training_script = ['imbalance']
'''
import warnings, datetime, uuid, os, json, shutil, pickle

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pandas as pd

print('installing package configuration')
os.system('pip3 install imbalanced-learn==0.5.0')
os.system('pip3 install scikit-learn==0.22.2.post1')
# os.system('pip3 install scipy==1.4.1')
# os.system('pip3 install scikit-learn==0.20.1')

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
# from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
		
'''
Taken from the example here:
https://imbalanced-learn.readthedocs.io/en/stable/

Plotting taken from:
https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html
'''

def train_imbalance(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

	# create file names
	model_name=common_name_model+'.pickle'
	files=list()
	RANDOM_STATE = 42

	if mtype=='c':

		# Create a pipeline
		pipeline = make_pipeline(NearMiss(version=2),
		                         LinearSVC(random_state=RANDOM_STATE))

		pipeline.fit(X_train, y_train)

		# Classify and report the results
		label_predictions = pipeline.predict(X_test)
		report=classification_report_imbalanced(y_test, label_predictions)
		print(report)
		accuracy=accuracy_score(y_test, label_predictions)
		print(accuracy)

		# now save the model in .pickle
		f=open(model_name,'wb')
		pickle.dump(pipeline, f)
		f.close()

	elif mtype == 'r':

		# Create a pipeline (usually works well with logistic regression 2 classes)
		pipeline = make_pipeline(SMOTE(random_state=RANDOM_STATE),
		                         LogisticRegression(random_state=0))

		pipeline.fit(X_train, y_train)

		# now save the model in .pickle
		f=open(model_name,'wb')
		pickle.dump(pipeline, f)
		f.close()
	
	# make sure to get files and model dir
	files.append(model_name)
	model_dir=os.getcwd()

	return model_name, model_dir, files
