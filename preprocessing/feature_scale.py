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

______                                           _             
| ___ \                                         (_)            
| |_/ / __ ___ _ __  _ __ ___   ___ ___  ___ ___ _ _ __   __ _ 
|  __/ '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
| |  | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
\_|  |_|  \___| .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, |
              | |                                         __/ |
              |_|                                        |___/ 
  ___  ______ _____ 
 / _ \ | ___ \_   _|
/ /_\ \| |_/ / | |  
|  _  ||  __/  | |  
| | | || |    _| |_ 
\_| |_/\_|    \___/ 

Scale features according to Allie's preprocessing API.
'''

import json, os, sys
import numpy as np 
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

def get_classes():
	count=2
	classes=list()
	while True:
		try:
			class_=sys.argv[count]
			classes.append(class_)
			print(classes)
			count=count+1
		except:
			break

	return classes

def feature_scale(feature_scaler, X_train, y_train):

	# more information about these scalers can be found @ 
	# https://scikit-learn.org/stable/modules/preprocessing.html

	if feature_scaler == 'binarizer':
		# scale the X values in the set 
		model = preprocessing.Binarizer()

	elif feature_scaler == 'one_hot_encoder':
		'''
		>>> enc.transform([['female', 'from US', 'uses Safari'],
			             	['male', 'from Europe', 'uses Safari']]).toarray()
			array([[1., 0., 0., 1., 0., 1.],
			       [0., 1., 1., 0., 0., 1.]])
		'''
		# This is on y values 
		model = preprocessing.OneHotEncoder(handle_unknown='ignore')

	elif feature_scaler == 'maxabs':
		model=preprocessing.MaxAbsScaler()

	elif feature_scaler == 'minmax':
		model=preprocessing.MinMaxScaler()

	elif feature_scaler == 'normalize':
		# L2 normalization
		model = preprocessing.Normalizer()

	elif feature_scaler == 'poly':
		# scale the X values in the set 
		model = PolynomialFeatures(2)

	elif feature_scaler == 'power_transformer':
		# scale the X values in the set 
		model = preprocessing.PowerTransformer(method='yeo-johnson')

	elif feature_scaler == 'quantile_transformer_normal':
		# scale the X values in the set 
		model = preprocessing.QuantileTransformer(output_distribution='normal')

	elif feature_scaler == 'robust':
		model=preprocessing.RobustScaler(quantile_range=(25, 75)) 

	elif feature_scaler == 'standard_scaler':
		# scale the X values in the set 
		model=preprocessing.StandardScaler()

	return model 
