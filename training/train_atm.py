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

Train models using ATM: https://github.com/HDI-Project/ATM

This is enabled if the default_training_script = ['atm']
'''
import pandas as pd
import os, sys, pickle, json, random, shutil, time
os.system('pip3 install atm==0.2.2')
os.system('pip3 install pandas==0.24.2')
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
                try:
                    data[feature_list[j]]=data[feature_list[j]]+[X_train[i][j]]
                except:
                    pass
            else:
                data[feature_list[j]]=[X_train[i][j]]
                print(data)

    data['class']=y_train
    data=pd.DataFrame(data, columns = list(data))
    
    return data

def train_atm(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):
    
    # create file names 
    model_name=common_name_model+'.pickle'
    csvname=common_name_model.split('_')[0]
    files=list()

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

    try:
        shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_train_transformed.csv',os.getcwd()+'/train.csv')
        shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_test_transformed.csv',os.getcwd()+'/test.csv')
    except:
        shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_train.csv',os.getcwd()+'/train.csv')
        shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_test.csv',os.getcwd()+'/test.csv')     
    
    # train models 
    results = atm.run(train_path='train.csv', class_column='class_')
    data_results_=str(results.describe())
    bestclassifier=str(results.get_best_classifier())
    scores=str(results.get_scores())

    # export classifier / transfer to model directory
    results.export_best_classifier(model_name, force=True)
    shutil.move(os.getcwd()+'/'+model_name, curdir+'/'+model_name)
    files.append('atm_temp')
    files.append(model_name)
    files.append('atm.db')
    os.chdir(curdir)
    model_dir=os.getcwd()

    return model_name, model_dir, files
