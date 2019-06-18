'''
ludwig_text
'''
import os, csv, json, random, sys, yaml, time, shutil
import numpy as np
from ludwig.api import LudwigModel

# write to .csv
def write_csv(filename, alldata, feature_labels, labels):

    with open(filename, mode='w') as csv_file:
        csv_writerfile = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # write all labels to csv 
        feature_labels.append('labels')
        csv_writerfile.writerow(feature_labels)
        for i in range(len(alldata)):
            # write all features to row 
            curlist=list(alldata[i])
            curlist=list(map(float,curlist))
            curlist.append(labels[i])
            csv_writerfile.writerow(curlist)

    return filename

def make_yaml(feature_labels, epochs):

    #  make the yaml file 
    # assume inputs in first column and outputs in second column 
    print('making yaml file --> model_definition.yaml')

    # assume everything that is not labels heading as a feature
    inputs='input_features:\n'
    for i in range(len(feature_labels)):
        if feature_labels[i] != 'labels':
            inputs=inputs+'    -\n        name: %s\n        type: %s\n'%(feature_labels[i], 'numerical')

    # assume everything in labels heading as a label 
    outputs='output_features:\n    -\n        name: %s\n        type: %s\n'%('labels', 'category')

    text=inputs+'\n'+outputs

    g=open('model_definition.yaml','w')
    g.write(text)
    g.close()

    return 'model_definition.yaml'

def train_ludwig(mtype, classes, jsonfile, alldata, labels, feature_labels, problemtype, default_features):
  
    jsonfilename='%s.json'%(jsonfile[0:-5]+"_ludwig_%s"%(default_features))
    filename='%s.csv'%(jsonfilename[0:-5])
    filename=write_csv(filename, alldata, feature_labels, labels)

    # now make a model_definition.yaml
    epochs=10
    feature_inputs=list()

    model_definition = make_yaml(feature_labels, epochs)
    print(os.getcwd())
    time.sleep(10)
    os.system('ludwig experiment --data_csv %s --model_definition_file model_definition.yaml --output_directory %s'%(filename, filename[0:-4]))
    os.rename('model_definition.yaml', filename[0:-4]+'.yaml')

    cur_dir2=os.getcwd()
    try:
        os.chdir(problemtype+'_models')
    except:
        os.mkdir(problemtype+'_models')
        os.chdir(problemtype+'_models')

    # now move all the files over to proper model directory 
    shutil.move(cur_dir2+'/'+jsonfilename, os.getcwd()+'/'+jsonfilename)
    shutil.move(cur_dir2+'/'+filename, os.getcwd()+'/'+filename)
    shutil.move(cur_dir2+'/'+jsonfilename[0:-5]+'.hdf5', os.getcwd()+'/'+jsonfilename[0:-5]+'.hdf5')
    shutil.move(cur_dir2+'/'+jsonfilename[0:-5]+'.yaml', os.getcwd()+'/'+jsonfilename[0:-5]+'.yaml')
    shutil.copytree(cur_dir2+'/'+jsonfilename[0:-5], os.getcwd()+'/'+jsonfilename[0:-5])
    shutil.rmtree(cur_dir2+'/'+jsonfilename[0:-5])

    return jsonfilename[0:-5], os.getcwd()