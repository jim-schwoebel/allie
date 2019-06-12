import os, sys, pickle, json, random, shutil, time 
import numpy as np
import matplotlib.pyplot as plt
import keras.models
from keras import layers 
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import datetime 

def train_keras(alldata, labels, mtype, jsonfile, problemtype, default_features):
    # get train and test data 
    start=time.time()
    x_train, x_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)
    modelname=jsonfile[0:-5]

    # MAKE MODEL (assume classification problem)
    ############################################################################
    model = Sequential()
    model.add(Dense(64, input_dim=len(x_train[0]), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=100,
              batch_size=128)

    # EVALUATE MODEL / PREDICT OUTPUT 
    ############################################################################
    score = model.evaluate(x_test, y_test, batch_size=128)

    print("\n final %s: %.2f%% \n" % (model.metrics_names[1], score[1]*100))
    print(model.predict(x_train[0][np.newaxis,:]))

    model = Sequential()
    model.add(Dense(64, input_dim=len(x_train[0]), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(np.array(list(x_train)+list(x_test)), np.array(list(y_train)+list(y_test)),epochs=100,batch_size=128)

    #SAVE TO DISK
    ############################################################################
    # serialize model to JSON
    model_json = model.to_json()
    with open(modelname+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(modelname+".h5")
    print("\n Saved %s.json model to disk"%(modelname))

    # SUMMARIZE RESULTS
    ############################################################################
    execution=time.time()-start
    print('summarizing data...')
    g=open(modelname+'.txt','w')
    g.write('SUMMARY OF MODEL')
    g.write('\n\n')
    g.write('Keras-based implementation of a neural network, 2 layers (relu | sigmoid activation functions), loss=binary_crossentropy, optimizer=rmsprop')
    g.write('\n\n')
    g.write('MODEL FILE NAME: \n\n %s.json | %s.h5'%(modelname,modelname))
    g.write('\n\n')
    g.write('DATE CREATED: \n\n %s'%(str(datetime.datetime.now())))
    g.write('\n\n')
    g.write('EXECUTION TIME: \n\n %s\n\n'%(str(execution)))
    g.write('GROUPS: \n\n')
    g.write('Group 1: %s (%s training, %s testing)'%(name1,str(int(len(x_train/2)),str(int(len(x_test)/2)))))
    g.write('\n')
    g.write('Group 2: %s (%s training, %s testing)'%(name2,str(int(len(y_train)/2)),str(int(len(y_test)/2))))
    g.write('\n\n')
    g.write('FEATURES: \n\n %s'%(default_features))
    g.write('\n\n')
    g.write('MODEL ACCURACY: \n\n')
    g.write('%s: %s \n\n'%(str('accuracy'),str(score[1]*100)))
    g.write('(C) 2019, NeuroLex Laboratories')
    g.close()


#####################

    jsonfilename='%s_keras.json'%(modelname)
    print('saving .JSON file (%s)'%(jsonfilename))
    jsonfile=open(jsonfilename,'w')
    
    data={'sample type': problemtype,
            'feature_set':default_features,
            'model name':jsonfilename[0:-5]+'.pickle',
            'accuracy':score[1],
            'model type':'keras_mlp',
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
    shutil.move(cur_dir2+'/'+modelname+".h5", os.getcwd()+'/'+modelname+".h5")
    shutil.move(cur_dir2+'/'+modelname+".json", os.getcwd()+'/'+modelname+".json")
    shutil.move(cur_dir2+'/'+modelname+".txt", os.getcwd()+'/'+modelname+".txt")