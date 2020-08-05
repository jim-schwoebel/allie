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

Train models using autokaggle: https://github.com/datamllab/autokaggle

This is enabled if the default_training_script = ['autokaggle']
'''
import os, pickle
curdir=os.getcwd()
print(os.getcwd())
print('initializing installation')
os.system('pip3 install autokaggle==0.1.0')
os.system('pip3 install scikit-learn==0.22')
from autokaggle.tabular_supervised import TabularClassifier
from autokaggle.tabular_supervised import TabularRegressor
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
