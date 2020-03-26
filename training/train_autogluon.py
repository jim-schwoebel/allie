from autogluon import TabularPrediction as task
import pandas as pd
import os, sys, pickle, json, random, shutil, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def convert_gluon(X_train, y_train):

    feature_list=list()
    for i in range(len(X_train[0])):
        feature_list.append('feature_'+str(i))

    feature_list.append('class')
    data=dict()

    for i in range(len(X_train)):
        for j in range(len(feature_list)-1):
            if i > 0:
                # print(data[feature_list[j]])
                try:
                    # print(feature_list[j])
                    # print(data)
                    # print(X_train[i][j])
                    # print(data[feature_list[j]])
                    # time.sleep(2)
                    data[feature_list[j]]=data[feature_list[j]]+[X_train[i][j]]
                except:
                    pass
                    # print(data)
                    # time.sleep(50)
                    # print(str(i)+'-i')
                    # print(j)

            else:
                data[feature_list[j]]=[X_train[i][j]]
                print(data)

    data['class']=y_train
    data=pd.DataFrame(data, columns = list(data))
    data=task.Dataset(data)

    return data

def train_autogluon(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):
    # get train and test data 
    X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)
    train_data = convert_gluon(X_train, y_train)
    test_data = convert_gluon(X_test, y_test)
    predictor = task.fit(train_data=train_data, label='class')
    accuracy = predictor.evaluate(test_data)
    print(accuracy)

    jsonfilename=jsonfile[0:-5]+'_'+str(default_features).replace("'",'').replace('"','')+'_autogluon.json'
    print('saving .JSON file (%s)'%(jsonfilename))
    jsonfile=open(jsonfilename,'w')
    if mtype in ['classification', 'c']:
        data={'sample type': problemtype,
            'feature_set':default_features,
            'model name':jsonfilename[0:-5]+'.pickle',
            'accuracy':accuracy,
            'model type':'AutoGluon_classification',
            'settings': settings,
        }
    elif mtype in ['regression', 'r']:
        data={'sample type': problemtype,
            'feature_set':default_features,
            'model name':jsonfilename[0:-5]+'.pickle',
            'accuracy':accuracy,
            'model type':'AutoGluon_regression',
            'settings': settings,
        }

    json.dump(data,jsonfile)
    jsonfile.close()

    # pickle store classifier
    f=open(jsonfilename[0:-5]+'.pickle','wb')
    pickle.dump(predictor, f)
    f.close()

    # now rename current directory with models (keep this info in a folder)
    newdir=jsonfilename[0:-5]
    os.mkdir(newdir)
    curdir=os.getcwd()
    shutil.copytree(curdir+'/dask-worker-space',newdir+'/dask-worker-space')
    shutil.copytree(curdir+'/AutogluonModels',newdir+'/AutoGluonModels')
    shutil.copytree(curdir+'/catboost_info',newdir+'/catboost_info')
    shutil.rmtree('dask-worker-space')
    shutil.rmtree('AutogluonModels')
    shutil.rmtree('catboost_info')

    cur_dir2=os.getcwd()
    try:
    	os.chdir(problemtype+'_models')
    except:
    	os.mkdir(problemtype+'_models')
    	os.chdir(problemtype+'_models')

    # now move all the files over to proper model directory 
    shutil.copytree(curdir+'/'+newdir, os.getcwd()+'/'+newdir)
    shutil.move(cur_dir2+'/'+jsonfilename, os.getcwd()+'/'+jsonfilename)
    shutil.move(cur_dir2+'/'+jsonfilename[0:-5]+'.pickle', os.getcwd()+'/'+jsonfilename[0:-5]+'.pickle')

    # remove temporary files
    try:
        shutil.rmtree(curdir+'/'+newdir)
    except:
        pass

    # get model_name 
    model_name=jsonfilename[0:-5]+'.pickle'
    model_dir=os.getcwd()

    return model_name, model_dir
