import os, sys, pickle, json, random, shutil, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from hpsklearn import HyperoptEstimator, any_preprocessing, any_classifier, any_regressor
from hyperopt import tpe
import numpy as np 

def train_hypsklearn(alldata, labels, mtype, jsonfile, problemtype, default_features):

    # get train and test data 
    X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)
    if mtype in [' classification', 'c']:

        modelname=jsonfile[0:-5]+'_hypsklearn_classification_%s'%(str(default_features).replace("'",'').replace('"',''))

        estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                                  preprocessing=any_preprocessing('my_pre'),
                                  algo=tpe.suggest,
                                  max_evals=100,
                                  trial_timeout=120)

        # Search the hyperparameter space based on the data
        estim.fit(X_train, y_train)



    elif mtype in ['regression','r']:

        modelname=jsonfile[0:-5]+'_hypsklearn_regression_%s'%(str(default_features).replace("'",'').replace('"',''))

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
    f=open(modelname+'.pickle','wb')
    pickle.dump(estim,f)
    f.close()

    jsonfilename='%s.json'%(modelname)
    print('saving .JSON file (%s)'%(jsonfilename))
    jsonfile=open(jsonfilename,'w')
    if mtype in ['classification', 'c']:
        data={'sample type': problemtype,
            'feature_set':default_features,
            'model name':jsonfilename[0:-5]+'.pickle',
            'accuracy':scores,
            'model type': bestmodel,
        }
    elif mtype in ['regression', 'r']:
        data={'sample type': problemtype,
            'feature_set':default_features,
            'model name':jsonfilename[0:-5]+'.pickle',
            'accuracy':scores,
            'model type': bestmodel,
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
    shutil.move(cur_dir2+'/'+modelname+'.json', os.getcwd()+'/'+modelname+'.json')
    shutil.move(cur_dir2+'/'+modelname+'.pickle', os.getcwd()+'/'+modelname+'.pickle')

    return modelname+'.pickle', os.getcwd()