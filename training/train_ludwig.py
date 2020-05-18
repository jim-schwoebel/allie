'''
ludwig_text
'''
import os, csv, json, random, sys, yaml, time, shutil
os.system('pip3 install tensorflow==1.15.2')
os.system('pip3 install ludwig==0.2.2.6')
from ludwig.api import LudwigModel
import pandas as pd
import numpy as np

def make_yaml(feature_labels, epochs):

    #  make the yaml file 
    # assume inputs in first column and outputs in second column 
    print('making yaml file --> model_definition.yaml')

    # assume everything that is not labels heading as a feature
    inputs='input_features:\n'
    for i in range(len(feature_labels)):
        if feature_labels[i] != 'class_':
            inputs=inputs+'    -\n        name: %s\n        type: %s\n'%(feature_labels[i], 'numerical')

    # assume everything in labels heading as a label 
    outputs='output_features:\n    -\n        name: %s\n        type: %s\n'%('class_', 'category')

    text=inputs+'\n'+outputs

    g=open('model_definition.yaml','w')
    g.write(text)
    g.close()

    return 'model_definition.yaml'

def train_ludwig(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_featurenames,transform_model,settings,model_session):

    # now make a model_definition.yaml
    model_name=common_name_model
    files=list()
    epochs=10
    feature_inputs=list()

    # get some random naming data
    curdir=os.getcwd()
    csvname=common_name_model.split('_')[0]

    # get training and testing data
    try:
        shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_train_transformed.csv',os.getcwd()+'/train.csv')
        shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_test_transformed.csv',os.getcwd()+'/test.csv')
    except:
        shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_train.csv',os.getcwd()+'/train.csv')  
        shutil.copy(curdir+'/'+model_session+'/data/'+csvname+'_test.csv',os.getcwd()+'/test.csv')
    
    # now read file to get features 
    data=pd.read_csv('train.csv')
    feature_labels=list(data)

    model_definition = make_yaml(feature_labels, epochs)
    print(os.getcwd())
    os.system('ludwig experiment --data_csv %s --model_definition_file model_definition.yaml --output_directory %s'%('train.csv', 'ludwig_files'))
    os.rename('model_definition.yaml', common_name_model+'.yaml')
    
    # add a bunch of files
    files.append('train.csv')
    files.append('test.csv')
    files.append('train.json')
    files.append('train.hdf5')
    files.append(common_name_model+'.yaml')
    files.append('ludwig_files')

    model_dir=os.getcwd()

    return model_name, model_dir, files