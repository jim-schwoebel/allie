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

Train models using autokeras: https://github.com/keras-team/autokeras

This is enabled if the default_training_script = ['autokeras']
'''
import os, pickle, json, shutil
import autokeras as ak
import tensorflow as tf
import numpy as np

'''
# plot model
https://autokeras.com/tutorial/structured_data_regression/
'''

def train_autokeras(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

	# create file names
	files=list()
	model_name=common_name_model

	# remove folder if it exists
	if mtype=='c':
		if 'structured_data_classifier' in os.listdir():
			shutil.rmtree('structured_data_classifier')
		model = ak.StructuredDataClassifier(max_trials=100)
		model.fit(X_train, y_train)
		files.append('structured_data_classifier')
	elif mtype == 'r':
		if 'structured_data_regressor' in os.listdir():
			shutil.rmtree('structured_data_regressor')
		model = ak.StructuredDataRegressor(max_trials=100)
		model.fit(X_train, y_train)
		files.append('structured_data_regressor')

	# show predictions
	predictions=model.predict(X_test).flatten()	
	print(predictions)

	model = model.export_model()
	print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>

	model.save(model_name+".h5")
	model_name=model_name+".h5"

	# get variables
	files.append(model_name)
	model_dir=os.getcwd()

	return model_name, model_dir, files
