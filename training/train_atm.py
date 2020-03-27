import pandas as pd
import os, sys, pickle, json, random, shutil, time
import numpy as np
from atm import ATM

def convert_(X_train, y_train):

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

    return data

def train_atm(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):

    # convert to proper format 
    all_data = convert_(alldata, labels)

    # create file names 
    jsonfilename=jsonfile[0:-5]+'_'+str(default_features).replace("'",'').replace('"','')+'_atm.json'
    csvfilename=jsonfilename[0:-5]+'.csv'
    picklefilename=jsonfilename[0:-5]+'.pickle'
    all_data.to_csv(csvfilename)

    # initialize and train classifier 
    atm = ATM()

    # create a temporary directory for all models 
    curdir=os.getcwd()
    try:
        os.mkdir('atm_temp')
        os.chdir('atm_temp')
    except:
        shutil.rmtree('atm_temp')
        os.mkdir('atm_temp')
        os.chdir('atm_temp')
        
    # train models 
    all_data.to_csv(csvfilename)
    results = atm.run(train_path=csvfilename)
    results.describe()
    bestclassifier=str(results.get_best_classifier())
    scores=str(results.get_scores())

    # export classifier / transfer to model directory
    results.export_best_classifier(picklefilename, force=True)
    shutil.copy(os.getcwd()+'/'+picklefilename, curdir+'/'+picklefilename)

    # go back out and remove temp directory
    os.chdir(curdir)
    shutil.rmtree('atm_temp')
    os.remove('atm.db')

    print('------------------------------------')
    print('          EXPORTING FILES           ')
    print('------------------------------------')

    # now export model 
    print('exporting .PICKLE model %s'%(picklefilename))
    

    # same json files 
    print('saving .JSON file (%s)'%(jsonfilename))
    jsonfile=open(jsonfilename,'w')
    if mtype in ['classification', 'c']:
        data={'sample type': problemtype,
            'feature_set':default_features,
            'model name':jsonfilename[0:-5]+'.pickle',
            'accuracy':bestclassifier,
            'model type':'ATM_classification - '+bestclassifier,
            'scores': scores,
            'settings': settings,
        }
    elif mtype in ['regression', 'r']:
        data={'sample type': problemtype,
            'feature_set':default_features,
            'model name':jsonfilename[0:-5]+'.pickle',
            'accuracy':bestclassifier,
            'model type':'ATM_regression - '+bestclassifier,
            'scores': scores,
            'settings': settings,
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
    shutil.move(cur_dir2+'/'+jsonfilename, os.getcwd()+'/'+jsonfilename)
    shutil.move(cur_dir2+'/'+jsonfilename[0:-5]+'.pickle', os.getcwd()+'/'+jsonfilename[0:-5]+'.pickle')
    shutil.move(cur_dir2+'/'+jsonfilename[0:-5]+'.csv', os.getcwd()+'/'+jsonfilename[0:-5]+'.csv')

    # remove temporary files
    try:
        shutil.rmtree(curdir+'/'+newdir)
    except:
        pass

    # get model_name 
    model_name=jsonfilename[0:-5]+'.pickle'
    model_dir=os.getcwd()

    return model_name, model_dir
