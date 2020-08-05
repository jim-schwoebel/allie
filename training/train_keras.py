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

Train models using keras simple MLP deep learning model: https://keras.io/getting-started/faq/

This is enabled if the default_training_script = ['keras']
'''
import os, sys, pickle, json, random, shutil, time, datetime, math 
import numpy as np
import matplotlib.pyplot as plt
import keras.models
from keras import layers 
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

def train_keras(X_train,X_test,y_train,y_test,mtype,common_name_model,problemtype,classes,default_features,transform_model,settings,model_session):

    files=list()
    model_name=common_name_model+".h5"
    
    # MAKE MODEL (assume classification problem)
    ############################################################################
    model = Sequential()
    model.add(Dense(64, input_dim=len(X_train[0]), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              epochs=100,
              batch_size=128)

    ############################################################################
    # serialize model to JSON
    # model_json = model.to_json()
    # with open(modelname+".json", "w") as json_file:
        # json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights(modelname+".h5")
    # print("\n Saved %s.json model to disk"%(modelname))

    # re-compile model
    # not to save optimizer variables in model data
    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'],
    )
    model.save(model_name)

    files.append(model_name)
    model_dir=os.getcwd()

    return model_name, model_dir, files
