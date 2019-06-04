'''
Train keras models. All of these are base model examples from the keras library.

Vision models

-->mnist_mlp.py Trains a simple deep multi-layer perceptron on the MNIST dataset.
-->mnist_cnn.py Trains a simple convnet on the MNIST dataset.
-->cifar10_cnn.py Trains a simple deep CNN on the CIFAR10 small images dataset.
-->cifar10_resnet.py Trains a ResNet on the CIFAR10 small images dataset.
-->conv_lstm.py Demonstrates the use of a convolutional LSTM network.
-->image_ocr.py Trains a convolutional stack followed by a recurrent stack and a CTC logloss function to perform optical character recognition (OCR).
-->mnist_acgan.py Implementation of AC-GAN (Auxiliary Classifier GAN) on the MNIST dataset
-->mnist_hierarchical_rnn.py Trains a Hierarchical RNN (HRNN) to classify MNIST digits.
-->mnist_siamese.py Trains a Siamese multi-layer perceptron on pairs of digits from the MNIST dataset.
-->mnist_swwae.py Trains a Stacked What-Where AutoEncoder built on residual blocks on the MNIST dataset.
-->mnist_transfer_cnn.py Transfer learning toy example.

Audio, Text & sequences examples

-->addition_rnn.py Implementation of sequence to sequence learning for performing addition of two numbers (as strings).
-->babi_rnn.py Trains a two-branch recurrent network on the bAbI dataset for reading comprehension.
-->babi_memnn.py Trains a memory network on the bAbI dataset for reading comprehension.
-->imdb_bidirectional_lstm.py Trains a Bidirectional LSTM on the IMDB sentiment classification task.
-->imdb_cnn.py Demonstrates the use of Convolution1D for text classification.
-->imdb_cnn_lstm.py Trains a convolutional stack followed by a recurrent stack network on the IMDB sentiment classification task.
-->imdb_fasttext.py Trains a FastText model on the IMDB sentiment classification task.
-->imdb_lstm.py Trains an LSTM model on the IMDB sentiment classification task.
-->lstm_stateful.py Demonstrates how to use stateful RNNs to model long sequences efficiently.
-->pretrained_word_embeddings.py Loads pre-trained word embeddings (GloVe embeddings) into a frozen Keras Embedding layer, and uses it to train a text classification model on the 20 Newsgroup dataset.
-->reuters_mlp.py Trains and evaluate a simple MLP on the Reuters newswire topic classification task.

REFERENCES

Base models: https://github.com/keras-team/keras/tree/master/exampleshttps://github.com/keras-team/keras/tree/master/examples

(1) https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/
(2) https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
(3) https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
(4) https://keras.io/getting-started/sequential-model-guide/
(5) https://keras.io/getting-started/functional-api-guide/

'''
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
import datetime 

# other stuff
import numpy as np 
import pandas, getpass, random, json, os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Embedding, LSTM, Bidirectional
from keras.layers import Activation, Conv1D, GlobalMaxPooling1D, recurrent, MaxPooling1D, GlobalAveragePooling1D
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from pandas import Series

# pre processing
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

''' BASELINE FUNCTIONS

# Define keras mosel types as various functions.

# This is what we will use to test various keras-based deep learning models for
performance later to see what architecture to look deeper at and optimize later.
'''

################################################### 
# BASELINE FUNCTIONS (for keras networks)

def get_features(directory, classname):
    # gets features from a prior directory featurized (makes more efficient)
    os.chdir(directory)
    listdir=os.listdir()
    features=list()
    labels=list()
    for i in range(len(listdir)):
        if listdir[i][-5:]=='.json':
            g=json.load(open(listdir[i]))
            feature=g['features']
            features.append(feature)
            labels.append(classname)           
    return features, labels
            
def transform_data(features, labels):
##    #audio features
    dimension=len(features[0])
##    ##########################################################################################################
##    ##                                      WHAT INPUT DATA LOOKS LIKE:                                     ##
##    ##########################################################################################################
##    ##    features = [-309.13900050822417, 62.994113729391806, -413.79101175121536, -126.44824022654625,    ##
##    ##                145.68530325221258, 36.63236047297892, 46.81091824978685, 224.72785358499007,         ##
##    ##                -37.22843646027476, 29.019041509665218, -132.72131653263105, 17.649062326512716,      ##
##    ##                ...                                                                                   ##
##    ##                -1.4978552705426866, 3.407496547510848, -5.401921783350719, 4.815302136763165,        ##
##    ##                -0.1618493623403922, 4.339649083555491, -7.4368054086188335, 6.86717455737755]        ##
##    ##                                                                                                      ##
##    ##    labels = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']            ##
##    ##########################################################################################################
##                                
##    # PREPROCESS THIS DATA TO PROPER FORMAT
##    feature_length=len(features[0])
##    new_features=list()
##    all_features=np.array([])
##    # get the min and max all feature matrices
##    print('appending all features into 1 numpy array')
##    for i in range(len(features)):
##        all_features=np.append(all_features,np.array(features[i]))
####    # check for list features to debug...
####    for i in range(len(all_features)):
####        if type(all_features[i])==type(list()):
####            print(i)
##
##    # normalize [0,1]
##    values = all_features.reshape((len(all_features), 1))
##    scaler = MinMaxScaler().fit(values)
##
##    print('looping through all features and making list')
##    for i in range(len(features)):
##        series = Series(features[i])
##        values= series.values
##        values = values.reshape((len(values), 1))
##        normalized = scaler.transform(values)
##        # one hot encoding via sklearn library
##        feature=np.concatenate(normalized)
##        new_features.append(feature)
        
##    features=new_features
    
    # integer encode
    values = np.array(labels)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    num_classes=np.amax(integer_encoded)+1
    new_labels=to_categorical(integer_encoded, num_classes)     
    labels=new_labels

    # ensure dimension match up!
    # now iterate through all labels and features to remove samples not of same length
    new_features=list()
    new_labels=list()
    print('iterating through features and making sure they are the same length')
    for i in range(len(features)):
        if len(features[i])==dimension:
            new_features.append(np.array(features[i]))
            new_labels.append(labels[i])
        else:
            pass

    features=np.array(new_features)
    labels=np.array(new_labels)
    
    ##########################################
    ##    DATA SHOULD NOW LOOK LIKE:        ##
    ##########################################
    ##    features =array([[0.65642969],    ##
    ##                     [0.64807976],    ##
    ##                     [0.65072932],    ##
    ##                     ...              ##
    ##                     [0.6576711 ]])   ##
    ##                                      ##
    ##    labels = array([[1., 0., 0.],     ##
    ##                    [1., 0., 0.],     ##
    ##                    [0., 0., 1.],     ##
    ##                    [1., 0., 0.],     ##
    ##                    [0., 1., 0.],     ##
    ##                    [0., 1., 0.],     ##
    ##                    [0., 0., 1.],     ##
    ##                    [1., 0., 0.],     ##
    ##                    [0., 0., 1.],     ##
    ##                    [0., 1., 0.]])    ##
    ##                                      ##
    ##########################################
        
    return features, labels


# We should number each of these deep learning frameworks in terms of keys
# either the problem is a multiple classification problem or binary problem.
# The only thing that changes really in these cases is the loss type
# (for multi-class problems = 'categorical_crossentropy', for binary='binary_crossentropy')

# we can name each model by model_1 and make the input as binary or multiple to tune the model 

# KEY PAIRING OF CURRENT MODELS:

# model_1 - Multi-Layer Perceptron (MLP), sequential model,
# model_2 - 

# as we build more models we can just added to binary_(N+1) or multi_(N+1) as a naming scheme

def save_model(model, model_dir, modelname):
    # model name should be 1, 2, 3, etc. initially before picking optimum model 
    # save model to disk 

    try:
        os.chdir(model_dir)
    except:
        os.mkdir(model_dir)
        os.chdir(model_dir)
    
    # serialize model to JSON
    model_json = model.to_json()
    jsonfilename=modelname+".json"
    h5filename=modelname+".h5"
    with open(jsonfilename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5filename)
    print("\n Saved %s.json model to disk"%(modelname))

    return jsonfilename, h5filename
    
def mlp_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes):

    # make a description of network type here
    space='\n\n'
    network_type = '2-layered multilayer perceptron'
    input_layer='input layer: ' + str(len(x_train[0]))+' numpy array represented as %s'%(str(x_train[0]))
    layer_1='1st layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layer_2= '2nd layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layers=layer_1+space+layer_2
    output_layer='output layer: ' + str(classes)+' represented as %s'%(str(y_test[0]))
    network_description=network_type+space+input_layer+space+layer_1+space+layer_2+space+output_layer
    print(network_description)
    
    # hyperparameters
    modelname=modelname+'_mlp'
    batch_size = 128
    epochs = 20
    num_classes=len(classes)

    model = Sequential()
    model.add(Dense(512, input_dim=len(x_train[0]),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='sigmoid'))

    model_summary=model.summary()

    if modeltype=='binary':
        # for binary classification 1 or 0 (output layer)
        loss='binary_crossentropy'
        model.compile(loss=loss,
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    elif modeltype=='multi':
        # for multiple classes [1,0,0] or [0,1,0]... etc.
        loss='categorical_crossentropy'
        model.compile(loss=loss,
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    elif modeltype=='regression':
        # for regression models
        loss='mse'
        model.compile(loss=loss,
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)

    loss=score[0]
    accuracy=score[1]

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    jsonfilename, h5filename = save_model(model, model_dir, modelname)

    data={
        'batch size':batch_size,
        'network type':network_type,
        'input_layer':input_layer,
        'intermediate layers':layers,
        'output_layer':output_layer,
        'network description':network_description,       
        'epochs':epochs,
        'classes':classes,
        'classnum':num_classes,
        'jsonfile':jsonfilename,
        'h5filename':h5filename,
        'x_train':x_train.tolist(),
        'y_train':y_train.tolist(),
        'x_test':x_test.tolist(),
        'y_test':y_test.tolist(),
        'datetime':str(datetime.datetime.now()),
        'accuracy':score[1],
        'loss':score[0],
        'losstype':loss,
        'modeltype':modeltype,
        'model summary':str(model_summary),
        'score':score,
        }

    print(type(model_summary))

    jsonfilename_2=modelname+'_data.json'
    jsonfile=open(jsonfilename_2,'w')
    json.dump(data,jsonfile)
    jsonfile.close()

    return loss, accuracy, jsonfilename_2



def bidir_lstm_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes):
    
    # make a description of network type here
    space='\n\n'
    network_type = 'bidirectional LSTM model'
    input_layer='input layer: ' + str(len(x_train[0]))+' numpy array represented as %s'%(str(x_train[0]))
    layer_1='1st layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layer_2= '2nd layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layers=layer_1+space+layer_2
    output_layer='output layer: ' + str(classes)+' represented as %s'%(str(y_test[0]))
    network_description=network_type+space+input_layer+space+layer_1+space+layer_2+space+output_layer
    print(network_description)
    
    # hyperparameters
    modelname=modelname+'_blstm'
    batch_size = 32
    maxlen=len(x_train[0])
    epochs = 20
    num_classes=len(classes)

    model = Sequential()
    model.add(Embedding(maxlen, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model_summary=model.summary()

    if modeltype=='binary':
        # for binary classification 1 or 0 (output layer)
        loss='binary_crossentropy'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    elif modeltype=='multi':
        # for multiple classes [1,0,0] or [0,1,0]... etc.
        loss='categorical_crossentropy'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    elif modeltype=='regression':
        # for regression models
        loss='mse'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[x_test, y_test])

    score = model.evaluate(x_test, y_test, verbose=0)

    loss=score[0]
    accuracy=score[1]

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    jsonfilename, h5filename = save_model(model, model_dir, modelname)

    data={
        'batch size':batch_size,
        'network type':network_type,
        'input_layer':input_layer,
        'intermediate layers':layers,
        'output_layer':output_layer,
        'network description':network_description,       
        'epochs':epochs,
        'classes':classes,
        'classnum':num_classes,
        'jsonfile':jsonfilename,
        'h5filename':h5filename,
        'x_train':x_train.tolist(),
        'y_train':y_train.tolist(),
        'x_test':x_test.tolist(),
        'y_test':y_test.tolist(),
        'datetime':str(datetime.datetime.now()),
        'accuracy':score[1],
        'loss':score[0],
        'losstype':loss,
        'modeltype':modeltype,
        'model summary':str(model_summary),
        'score':score,
        }

    print(type(model_summary))

    jsonfilename_2=modelname+'_data.json'
    jsonfile=open(jsonfilename_2,'w')
    json.dump(data,jsonfile)
    jsonfile.close()

    return loss, accuracy, jsonfilename_2


def cnn_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes):
    
    # make a description of network type here
    space='\n\n'
    network_type = 'bidirectional LSTM model'
    input_layer='input layer: ' + str(len(x_train[0]))+' numpy array represented as %s'%(str(x_train[0]))
    layer_1='1st layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layer_2= '2nd layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layers=layer_1+space+layer_2
    output_layer='output layer: ' + str(classes)+' represented as %s'%(str(y_test[0]))
    network_description=network_type+space+input_layer+space+layer_1+space+layer_2+space+output_layer
    print(network_description)
    
    # hyperparameters
    modelname=modelname+'_cnn'
    max_features=len(x_train[0])
    batch_size = 32
    embedding_dims = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 2
    maxlen=len(x_train[0])
    num_classes=len(classes)

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    model_summary=model.summary()

    if modeltype=='binary':
        # for binary classification 1 or 0 (output layer)
        loss='binary_crossentropy'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    elif modeltype=='multi':
        # for multiple classes [1,0,0] or [0,1,0]... etc.
        loss='categorical_crossentropy'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    elif modeltype=='regression':
        # for regression models
        loss='mse'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[x_test, y_test])

    score = model.evaluate(x_test, y_test, verbose=0)

    loss=score[0]
    accuracy=score[1]

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    jsonfilename, h5filename = save_model(model, model_dir, modelname)

    data={
        'batch size':batch_size,
        'network type':network_type,
        'input_layer':input_layer,
        'intermediate layers':layers,
        'output_layer':output_layer,
        'network description':network_description,       
        'epochs':epochs,
        'classes':classes,
        'classnum':num_classes,
        'jsonfile':jsonfilename,
        'h5filename':h5filename,
        'x_train':x_train.tolist(),
        'y_train':y_train.tolist(),
        'x_test':x_test.tolist(),
        'y_test':y_test.tolist(),
        'datetime':str(datetime.datetime.now()),
        'accuracy':score[1],
        'loss':score[0],
        'losstype':loss,
        'modeltype':modeltype,
        'model summary':str(model_summary),
        'score':score,
        }

    print(type(model_summary))

    jsonfilename_2=modelname+'_data.json'
    jsonfile=open(jsonfilename_2,'w')
    json.dump(data,jsonfile)
    jsonfile.close()

    return loss, accuracy, jsonfilename_2


def rnn_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes):

    # need 2 joint variables dependent and then an output class 
    
    # make a description of network type here
    space='\n\n'
    network_type = 'bidirectional LSTM model'
    input_layer='input layer: ' + str(len(x_train[0]))+' numpy array represented as %s'%(str(x_train[0]))
    layer_1='1st layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layer_2= '2nd layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layers_total=layer_1+space+layer_2
    output_layer='output layer: ' + str(classes)+' represented as %s'%(str(y_test[0]))
    network_description=network_type+space+input_layer+space+layers_total+space+output_layer
    print(network_description)
    
    # hyperparameters
    modelname=modelname+'_rnn'
    maxlen=len(x_train[0])

    num_classes=len(classes)
    RNN = recurrent.LSTM
    EMBED_HIDDEN_SIZE = 50
    SENT_HIDDEN_SIZE = 100
    QUERY_HIDDEN_SIZE = 100
    BATCH_SIZE = 32
    EPOCHS = 40

    model = Sequential()

    sentence = layers.Input(shape=(x_train[0].shape), dtype='float32')
    encoded_sentence = layers.Embedding(maxlen, EMBED_HIDDEN_SIZE)(sentence)
    encoded_sentence = layers.Dropout(0.3)(encoded_sentence)

    question = layers.Input(shape=(y_train[0].shape), dtype='int32')
    encoded_question = layers.Embedding(num_classes, EMBED_HIDDEN_SIZE)(question)
    encoded_question = layers.Dropout(0.3)(encoded_question)
    encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
    encoded_question = layers.RepeatVector(maxlen)(encoded_question)

    merged = layers.add([encoded_sentence, encoded_question])
    merged = RNN(EMBED_HIDDEN_SIZE)(merged)
    merged = layers.Dropout(0.3)(merged)
    preds = layers.Dense(num_classes, activation='sigmoid')(merged)
    model = Model([sentence, question], preds)

    model_summary=model.summary()

    if modeltype=='binary':
        # for binary classification 1 or 0 (output layer)
        loss='binary_crossentropy'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    elif modeltype=='multi':
        # for multiple classes [1,0,0] or [0,1,0]... etc.
        loss='categorical_crossentropy'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    elif modeltype=='regression':
        # for regression models
        loss='mse'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])


    model.fit([x_train,y_train], y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.05)

    score = model.evaluate([x_test,y_test], y_test, verbose=0)

    loss=score[0]
    accuracy=score[1]

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    jsonfilename, h5filename = save_model(model, model_dir, modelname)

    data={
        'batch size':batch_size,
        'network type':network_type,
        'input_layer':input_layer,
        'intermediate layers':layers_total,
        'output_layer':output_layer,
        'network description':network_description,       
        'epochs':epochs,
        'classes':classes,
        'classnum':num_classes,
        'jsonfile':jsonfilename,
        'h5filename':h5filename,
        'x_train':x_train.tolist(),
        'y_train':y_train.tolist(),
        'x_test':x_test.tolist(),
        'y_test':y_test.tolist(),
        'datetime':str(datetime.datetime.now()),
        'accuracy':score[1],
        'loss':score[0],
        'losstype':loss,
        'modeltype':modeltype,
        'model summary':str(model_summary),
        'score':score,
        }

    print(type(model_summary))

    jsonfilename_2=modelname+'_data.json'
    jsonfile=open(jsonfilename_2,'w')
    json.dump(data,jsonfile)
    jsonfile.close()

    return loss, accuracy, jsonfilename_2


def cnnlstm_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes):

    # make a description of network type here
    space='\n\n'
    network_type = 'bidirectional LSTM model'
    input_layer='input layer: ' + str(len(x_train[0]))+' numpy array represented as %s'%(str(x_train[0]))
    layer_1='1st layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layer_2= '2nd layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layers_total=layer_1+space+layer_2
    output_layer='output layer: ' + str(classes)+' represented as %s'%(str(y_test[0]))
    network_description=network_type+space+input_layer+space+layers_total+space+output_layer
    print(network_description)
    
    # embedding hyperparameters 
    modelname=modelname+'_cnn_lstm'
    max_features = 20000
    maxlen=len(x_train[0])
    num_classes=len(classes)
    embedding_size = 128
    num_classes=len(classes)

    # convolution 
    kernel_size = 5
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 70

    # Training (2 epochs is enough, highly sensitive) 
    batch_size = 30
    epochs = 2

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(num_classes))
    # activation is sigmoid if 2 classes
    model.add(Activation('sigmoid'))

    model_summary=model.summary()

    if modeltype=='binary':
        # for binary classification 1 or 0 (output layer)
        loss='binary_crossentropy'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    elif modeltype=='multi':
        # for multiple classes [1,0,0] or [0,1,0]... etc.
        loss='categorical_crossentropy'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    elif modeltype=='regression':
        # for regression models
        
        loss='mse'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    loss_, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('Test loss:', loss_)
    print('Test accuracy:', acc)

    jsonfilename, h5filename = save_model(model, model_dir, modelname)

    data={
        'batch size':batch_size,
        'network type':network_type,
        'input_layer':input_layer,
        'intermediate layers':layers_total,
        'output_layer':output_layer,
        'network description':network_description,       
        'epochs':epochs,
        'classes':classes,
        'classnum':num_classes,
        'jsonfile':jsonfilename,
        'h5filename':h5filename,
        'x_train':x_train.tolist(),
        'y_train':y_train.tolist(),
        'x_test':x_test.tolist(),
        'y_test':y_test.tolist(),
        'datetime':str(datetime.datetime.now()),
        'accuracy':acc,
        'loss':loss_,
        'losstype':loss,
        'modeltype':modeltype,
        'model summary':str(model_summary),
        }

    print(type(model_summary))

    jsonfilename_2=modelname+'_data.json'
    jsonfile=open(jsonfilename_2,'w')
    json.dump(data,jsonfile)
    jsonfile.close()

    return loss_, acc, jsonfilename_2

def fasttext_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes):

    # make a description of network type here
    space='\n\n'
    network_type = 'bidirectional LSTM model'
    input_layer='input layer: ' + str(len(x_train[0]))+' numpy array represented as %s'%(str(x_train[0]))
    layer_1='1st layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layer_2= '2nd layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layers_total=layer_1+space+layer_2
    output_layer='output layer: ' + str(classes)+' represented as %s'%(str(y_test[0]))
    network_description=network_type+space+input_layer+space+layers_total+space+output_layer
    print(network_description)

    # define hyperparameters
    modelname=modelname+'_fasttext'
    max_features = 20000
    maxlen = len(x_train[0])
    batch_size = 32
    embedding_dims = 50
    epochs = 5
    num_classes=len(classes)
    
    # build model 
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))

    # we add a GlobalAveragePooling1D, which will average the embeddings
    # of all words in the document
    model.add(GlobalAveragePooling1D())

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(num_classes, activation='sigmoid'))

    model_summary=model.summary()

    if modeltype=='binary':
        # for binary classification 1 or 0 (output layer)
        loss='binary_crossentropy'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    elif modeltype=='multi':
        # for multiple classes [1,0,0] or [0,1,0]... etc.
        loss='categorical_crossentropy'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    elif modeltype=='regression':
        # for regression models
        
        loss='mse'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    loss_, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('Test loss:', loss_)
    print('Test accuracy:', acc)

    jsonfilename, h5filename = save_model(model, model_dir, modelname)

    data={
        'batch size':batch_size,
        'network type':network_type,
        'input_layer':input_layer,
        'intermediate layers':layers_total,
        'output_layer':output_layer,
        'network description':network_description,       
        'epochs':epochs,
        'classes':classes,
        'classnum':num_classes,
        'jsonfile':jsonfilename,
        'h5filename':h5filename,
        'x_train':x_train.tolist(),
        'y_train':y_train.tolist(),
        'x_test':x_test.tolist(),
        'y_test':y_test.tolist(),
        'datetime':str(datetime.datetime.now()),
        'accuracy':acc,
        'loss':loss_,
        'losstype':loss,
        'modeltype':modeltype,
        'model summary':str(model_summary),
        }

    jsonfilename_2=modelname+'_data.json'
    jsonfile=open(jsonfilename_2,'w')
    json.dump(data,jsonfile)
    jsonfile.close()

    return loss_, acc, jsonfilename_2

def lstm_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes):

    # make a description of network type here
    space='\n\n'
    network_type = 'bidirectional LSTM model'
    input_layer='input layer: ' + str(len(x_train[0]))+' numpy array represented as %s'%(str(x_train[0]))
    layer_1='1st layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layer_2= '2nd layer: 512 input layer, relu activation, 0.20 dropout regularization'
    layers_total=layer_1+space+layer_2
    output_layer='output layer: ' + str(classes)+' represented as %s'%(str(y_test[0]))
    network_description=network_type+space+input_layer+space+layers_total+space+output_layer
    print(network_description)

    # hyperparameters
    modelname=modelname+'_lstm'
    max_features = 20000
    maxlen = len(x_train[0])  
    batch_size = 32
    class_num=len(classes)
    num_classes=class_num
    epochs=15

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(class_num, activation='sigmoid'))

    model_summary=model.summary()

    # network definition
    if modeltype=='binary':
        # for binary classification 1 or 0 (output layer)
        loss='binary_crossentropy'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    elif modeltype=='multi':
        # for multiple classes [1,0,0] or [0,1,0]... etc.
        loss='categorical_crossentropy'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    elif modeltype=='regression':
        # for regression models
        
        loss='mse'
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    loss_, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    
    print('Test loss:', loss_)
    print('Test accuracy:', acc)

    jsonfilename, h5filename = save_model(model, model_dir, modelname)

    data={
        'batch size':batch_size,
        'network type':network_type,
        'input_layer':input_layer,
        'intermediate layers':layers_total,
        'output_layer':output_layer,
        'network description':network_description,       
        'epochs':epochs,
        'classes':classes,
        'classnum':num_classes,
        'jsonfile':jsonfilename,
        'h5filename':h5filename,
        'x_train':x_train.tolist(),
        'y_train':y_train.tolist(),
        'x_test':x_test.tolist(),
        'y_test':y_test.tolist(),
        'datetime':str(datetime.datetime.now()),
        'accuracy':acc,
        'loss':loss_,
        'losstype':loss,
        'modeltype':modeltype,
        'model summary':str(model_summary),
        }

    jsonfilename_2=modelname+'_data.json'
    jsonfile=open(jsonfilename_2,'w')
    json.dump(data,jsonfile)
    jsonfile.close()

    return loss_, acc, jsonfilename_2

###################################################
    
'''
DATA PREPROCESSING

# convert categorical data to numbers
# create some key pairing to reverse engineer classes when loading models (for multi-class support)
# store this data in a temp .json file (in case things crash)

'''
# load some dummy data here
# labels = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']

# can read the features of the folder
hostdir='/Users/'+getpass.getuser()+'/nlx-model/nlx-audiomodel/'
foldernum=int(input('how many classes do you want to train?'))
classes=list()
for i in range(foldernum):
    classes.append(input('what is the name of class %s?'%(str(i+1))))

jsonfilename=''
for i in range(len(classes)):
    if i == 0:
        jsonfilename=classes[i]
    else:
        jsonfilename=jsonfilename+'_'+classes[i]

modelname=jsonfilename
jsonfilename=jsonfilename+'.json'

#get classes
totals=list()
for i in range(len(classes)):
    features=list()
    labels=list()
    directory=hostdir+classes[i]
    features, labels = get_features(directory, classes[i])
    total=[features,labels]
    totals.append(total)
    
# now balance datasets (in terms of labels) - take from prior scripts
total_lengths=list()
for i in range(len(totals)):
    total_lengths.append(len(totals[i][0]))
    
min_num=np.amin(total_lengths)

for i in range(len(totals)):
    while min_num != len(totals[i][0]):
        totals[i][0].pop()
        totals[i][1].pop()

new_features=list()
new_labels=list()
for i in range(len(totals)):
    new_features=new_features + totals[i][0]
    new_labels=new_labels + totals[i][1]

# now shuffle datasets
features=new_features
labels=new_labels

# we can write this into .json as well...
os.chdir('/Users/'+getpass.getuser()+'/nlx-model/nlx-audiomodelkeras/')
model_dir='/Users/'+getpass.getuser()+'/nlx-model/nlx-audiomodelkeras/models/'

data = {
    'features':features,
    'labels':labels,
    }
jsonfile=open(jsonfilename,'w')
json.dump(data,jsonfile)
jsonfile.close()

# note that the features should be scaled from [0,1], but we can always
# change this pre-processing step into the future 
[features2, labels2] = transform_data(features,labels)

x_train, x_test, y_train, y_test = train_test_split(features2, labels2, test_size=0.33, random_state=42)

x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

###################################################

# DATA MODELING (WRITE ENCODER TO .JSON LATER)

# create various combinations of these variables to test as an input to the next function...
# ---> use itertools...

# now create combinations of various configurations to batch process 
modeltype='multi'

# OPERATIONAL

[mlp_loss, mlp_accuracy, mlp_filename]=mlp_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes)
print(mlp_accuracy)
#[lstm_loss,lstm_accuracy,lstm_filename]=lstm_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes)
#[blstm_loss, blstm_accuracy, blstm_filename]=bidir_lstm_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes)
#[cnn_loss, cnn_accuracy, cnn_filename]=cnn_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes)
#[cnnlstm_loss, cnnlstm_accuracy, cnnlstm_filename]=cnnlstm_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes)
#[ftext_loss,ftext_accuracy,ftext_filename]=fasttext_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes)

'''
Implmented all the relevant test examples in the keras library.

ALREADY IMPLEMENTED
-->reuters_mlp.py Trains and evaluate a simple MLP on the Reuters newswire topic classification task.
-->imdb_bidirectional_lstm.py Trains a Bidirectional LSTM on the IMDB sentiment classification task.
-->imdb_cnn.py Demonstrates the use of Convolution1D for text classification.
-->imdb_cnn_lstm.py Trains a convolutional stack followed by a recurrent stack network on the IMDB sentiment classification task.
-->imdb_fasttext.py Trains a FastText model on the IMDB sentiment classification task.
-->imdb_lstm.py Trains an LSTM model on the IMDB sentiment classification task.

TROUBLE IMPLEMENTING
--> [rnn_loss, rnn_accuracy, rnn_filename]=rnn_model(x_train, y_train, x_test, y_test, modelname, modeltype, model_dir, classes)
-->babi_memnn.py Trains a memory network on the bAbI dataset for reading comprehension.
-->pretrained_word_embeddings.py Loads pre-trained word embeddings (GloVe embeddings) into a frozen Keras Embedding layer, and uses it to train a text classification model on the 20 Newsgroup dataset.
'''

################################################################################
# OPTIMIZE MODEL (HYPERPARAMETERS)
################################################################################

# now go through all the typical network configurations and choose the most
# optimized network configuration

# variables to tune (4 each category)
epochs=[25,50,100,200]
batch_size=[32, 64, 128, 256]
embedding_sizes=[64, 128, 256, 512]
layer_nums=[2,4,8,16]
activation_types=['relu','softmax','sigmoid']
#only use sigmoid if binary classification 
optimizer_types=['rmsprop','adam']

# now pick the most accurate model
accuracies=[mlp_accuracy,lstm_accuracy,blstm_accuracy,cnn_accuracy,cnnlstm_accuracy, ftext_accuracy]
filenames_list=[mlp_filename,lstm_filename,blstm_filename,cnn_filename,ftext_filename]
maxval=np.amax(accuracies)
index=accuracies.index(maxval)

if index==0:
    filename=mlp_filename
elif index==1:
    filename=lstm_filename
elif index==2:
    filename=blstm_filename
elif index==4:
    filename=cnn_filename
elif index==5:
    filename=ftext_filename

##for i in range(len(filenames_list)):
##    if filenames_list[i] != filename:
##        os.remove(filename

# filename now can be stored and all others to be deleted 

# delete all the models and only keep the most accurate or summarize session

# # # # # FUTURE # # # # # # # # #
