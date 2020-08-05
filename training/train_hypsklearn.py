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

Train models using hypsklearn: https://github.com/hyperopt/hyperopt-sklearn

This is enabled if the default_training_script = ['hypsklearn']
'''
import os, pickle
os.system('export OMP_NUM_THREADS=1')
from hpsklearn import HyperoptEstimator, any_preprocessing, any_classifier, any_regressor
from hyperopt import tpe

def train_hypsklearn(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

    modelname=common_name_model+'.pickle'
    files=list()
    
    if mtype in [' classification', 'c']:

        estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                                  preprocessing=any_preprocessing('my_pre'),
                                  algo=tpe.suggest,
                                  max_evals=100,
                                  trial_timeout=120)

        # Search the hyperparameter space based on the data
        estim.fit(X_train, y_train)



    elif mtype in ['regression','r']:

        estim = HyperoptEstimator(classifier=any_regressor('my_clf'),
                                  preprocessing=any_preprocessing('my_pre'),
                                  algo=tpe.suggest,
                                  max_evals=100,
                                  trial_timeout=120)

        # Search the hyperparameter space based on the data

        estim.fit(X_train, y_train)

    # Show the results
    print(estim.score(X_test, y_test))
    print(estim.best_model())
    scores=estim.score(X_test, y_test)
    bestmodel=str(estim.best_model())

    print('saving classifier to disk')
    f=open(modelname,'wb')
    pickle.dump(estim,f)
    f.close()

    files.append(modelname)
    modeldir=os.getcwd()

    return modelname, modeldir, files
