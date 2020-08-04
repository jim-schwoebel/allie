import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
import autosklearn.classification as asklc
import sklearn.metrics
import os, shutil

def train_autosklearn(alldata, labels, mtype, jsonfile, problemtype, default_features):

    foldername=jsonfile[0:-5]+'_autosklearn_%s'%(default_features)
    X_train, X_test, y_train, y_test = train_test_split(alldata, 
                                                        labels,
                                                        train_size=0.750,
                                                        test_size=0.250,
                                                        random_state=42,
                                                        shuffle=True)
    feature_types = (['numerical'] * len(X_train[0]))

    automl = asklc.AutoSklearnClassifier(
        time_left_for_this_task=60,
        per_run_time_limit=300,
        ml_memory_limit=10240,
        tmp_folder=os.getcwd()+'/'+foldername+'_tmp',
        output_folder=os.getcwd()+'/'+foldername,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False)

    automl.fit(X_train, 
               y_train,
               dataset_name=jsonfile[0:-5],
               feat_type=feature_types)

    y_predictions = automl.predict(X_test)
    acc= sklearn.metrics.accuracy_score(y_true=y_test,
                                         y_pred=y_predictions)
    print("Accuracy:", acc)

    print('saving classifier to disk')
    f=open(modelname+'.pickle','wb')
    pickle.dump(automl,f)
    f.close()

    data={'sample type': problemtype,
        'training script': 'autosklearn',
        'feature_set':default_features,
        'model name':modelname+'.pickle',
        'accuracy':acc,
        'model type':'sc_'+classifiername,
        }

    g2=open(modelname+'.json','w')
    json.dump(data,g2)
    g2.close()

    cur_dir2=os.getcwd()
    try:
        os.chdir(problemtype+'_models')
    except:
        os.mkdir(problemtype+'_models')
        os.chdir(problemtype+'_models')

    # now move all the files over to proper model directory 
    shutil.move(cur_dir2+'/'+modelname+'.json', os.getcwd()+'/'+modelname+'.json')
    shutil.move(cur_dir2+'/'+modelname+'.pickle', os.getcwd()+'/'+modelname+'.pickle')

