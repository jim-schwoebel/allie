'''
fingerprint audio models in a streaming folder. 

first determine the file types available in the folder + load appropriate featurizers, as necessary.

change to load directory 
'''
import os, json 
import os, getpass
import keras.models
from keras import layers 
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout
import numpy as np
import time, pickle 

##########################################################
##                  HELPER FUNCTIONS                    ##
##########################################################

def prev_dir(directory):
    g=directory.split('/')
    dir_=''
    for i in range(len(g)):
        if i != len(g)-1:
            if i==0:
                dir_=dir_+g[i]
            else:
                dir_=dir_+'/'+g[i]
    # print(dir_)
    return dir_

def classifyfolder(listdir):
    filetypes=list()
    for i in range(len(listdir)):
        if listdir[i].endswith(('.mp3', '.wav')):
            filetypes.append('audio')
        elif listdir[i].endswith(('.png', '.jpg')):
            filetypes.append('image')
        elif listdir[i].endswith(('.txt')):
            filetypes.append('text')
        elif listdir[i].endswith(('.mp4', '.avi')):
            filetypes.append('video')
        elif listdir[i].endswith(('.csv')):
            filetypes.append('csv')

    counts={'audio': filetypes.count('audio'),
            'image': filetypes.count('image'),
            'text': filetypes.count('text'),
            'video': filetypes.count('video'),
            'csv': filetypes.count('.csv')}

    # get back the type of folder (main file type)
    totalfiles=sum(counts.values())
    values=list(counts.values())
    indices=list()

    for i in range(len(values)):
        if values[i] > 0:
            indices.append(i)

    # now that we have all the right indices, we can now output all the right 
    labels=list(counts)
    sampletypes=list()
    for i in range(len(indices)):
        sampletypes.append(labels[indices[i]])

    return totalfiles, sampletypes, counts

def rename_files():
    # remove json files 
    listdir=os.listdir() 
    for i in range(len(listdir)):
        if listdir[i][-5:]=='.json':
            os.remove(listdir[i])

    # now rename files 
    listdir=os.listdir()
    for i in range(len(listdir)):
        if listdir[i].endswith(('.mp3', '.wav')):
            os.rename(listdir[i], str(i)+'_audio'+listdir[i][-4:])
        elif listdir[i].endswith(('.png', '.jpg')):
            os.rename(listdir[i], str(i)+'_image'+listdir[i][-4:])
        elif listdir[i].endswith(('.txt')):
            os.rename(listdir[i], str(i)+'_text'+listdir[i][-4:])
        elif listdir[i].endswith(('.mp4', '.avi')):
            os.rename(listdir[i], str(i)+'_video'+listdir[i][-4:])
        elif listdir[i].endswith(('.csv')):
            os.rename(listdir[i], str(i)+'_csv'+listdir[i][-4:])

def detect_models():
    # takes in current directory and detects models

    listdir=os.listdir()

    # make a list to detect models 
    tpot_models=list()
    hypsklearn_models=list()
    scsr_models=list()
    devol_models=list()
    keras_models=list()
    ludwig_models=list()

    # get a list of files for models 
    for i in range(len(listdir)):
        data=dict()
        if listdir[i].find('tpot')>0 and listdir[i][-7:]=='.pickle':
            data[listdir[i]]=json.load(open(listdir[i][0:-7]+'.json'))
            tpot_models.append(data)
        elif listdir[i].find('sc')>0 and listdir[i][-7:]=='.pickle':
            data[listdir[i]]=json.load(open(listdir[i][0:-7]+'.json'))
            scsr_models.append(data)
        elif listdir[i].find('sr')>0 and listdir[i][-7:]=='.pickle':
            data[listdir[i]]=json.load(open(listdir[i][0:-7]+'.json'))
            scsr_models.append(data)
        elif listdir[i].find('hypsklearn')>0 and listdir[i][-7:]=='.pickle':
            data[listdir[i]]=json.load(open(listdir[i][0:-7]+'.json'))
            hypsklearn_models.append(data)
        elif listdir[i].find('keras')>0 and listdir[i][-3:]=='.h5':
            data[listdir[i]]=json.load(open(listdir[i][0:-7]+'.json'))
            keras_models.append(data)
        elif listdir[i].find('ludwig') and listdir[i].find('.')<0:
            data[listdir[i]]=json.load(open(listdir[i][0:-7]+'.json'))
            ludwig_models.append(data)

    # make a dictionary outputting all models
    data={'tpot_models': tpot_models,
          'scsr_models': scsr_models,
          'devol_models': devol_models,
          'keras_models': keras_models,
          'ludwig_models': ludwig_models,
         }
    # print(data)

    return data 

def load_tpot():
    # load the TPOT file 
    # make a prediction 
    return

def load_hypsklearn():
    # load the hypsklearn pickle file 
    # make a prediction 
    return

def load_scsr():
    # load the pickle file (or joblib if compressed)
    # make a prediction 
    return 

def load_devol():
     # load in the model file (compressed or not)
     return 

def load_keras():
    # load in the model file (compressed or not)
    return

def load_ludwig():
    # make a .CSV output of all the features in the folder 
    # make a ludwig prediction
    # load that output prediction 
    return 

def model_schema():
    models={'audio': dict(),
            'text': dict(),
            'image': dict(),
            'video': dict(),
            'csv': dict()
            }
    return models 

def make_predictions(sampletype, feature_set, model_dir, load_dir):
    '''
    Take in a sampletype (e.g. 'audio'), feature set (e.g. 'librosa_features'), the model
    directory ('audio_models'), and a load directory and update the .JSON file databases with 
    features with model predictions. 
    '''

    # detect machine learning models
    model_data=detect_models()
    # 'tpot_models': [], 'scsr_models': [], 'devol_models': [], 'keras_models': [], 'ludwig_models': []
    model_list=list(model_data)

    # now get all appropriate files 
    os.chdir(load_dir)
    listdir=os.listdir()
    jsonfilelist=list()

    for i in range(len(listdir)):
        if listdir[i][-5:]=='.json':
            g=json.load(open(listdir[i]))
            if g['sampletype'] == sampletype:
                jsonfilelist.append(listdir[i])

    print(jsonfilelist)
    for i in range(len(model_list)):
        if model_data[model_list[i]] != []:

            print(model_list[i])
            temp_models=list(model_data[model_list[i]])
            for j in range(len(temp_models)):

                modelname=list(temp_models[j])[0]
                print(modelname)
                time.sleep(2)
                if modelname.endswith('.pickle'):

                    # load model 
                    os.chdir(model_dir)
                    print(os.getcwd())
                    loadmodel=open(modelname, 'rb')
                    model = pickle.load(loadmodel)
                    loadmodel.close()

                    # modeldata
                    modeldata=json.load(open(modelname[0:-7]+'.json'))

                    # now only load featurized samples and make predictions 
                    for k in range(len(jsonfilelist)):
                        # load directory 
                        os.chdir(load_dir)

                        # get features (from prior array)
                        jsonfile=json.load(open(jsonfilelist[k]))
                        features=jsonfile['features'][sampletype][feature_set]['features']
                        features=np.array(features).reshape(1,-1)

                        # predict model 
                        output=str(model.predict(features)[0])
                        print(output)

                        # now get the actual class from the classifier name (assume 2 classes)
                        one=modelname.split('_')[0]
                        two=modelname.split('_')[1]

                        if output == '0':
                            class_=one
                        elif output == '1':
                            class_=two

                        # now update the database 
                        try:
                            models=jsonfile['models']
                        except:
                            models=model_schema()

                        temp=models[sampletype]
                        temp[class_]= modeldata
                        models[sampletype]=temp
                        jsonfile['models']=models

                        jsonfilename=open(jsonfilelist[k],'w')
                        json.dump(jsonfile,jsonfilename)
                        jsonfilename.close() 


                elif modelname.endswith('.h5'):
                    # load h5 model 
                    loaded_model.load_weights(modelname+".h5")
                    output=int(loaded_model.predict_classes(sample[np.newaxis,:]))

##########################################################
##                     CLEAN FOLDER                    ##
##########################################################

# load the default feature set 
cur_dir = os.getcwd()
prevdir= prev_dir(cur_dir)
os.chdir(prevdir+'/load_dir')
rename_files()

# get all the default feature arrays 
settings=json.load(open(prevdir+'/settings.json'))
default_audio_features=settings['default_audio_features']
default_text_features=settings['default_text_features']
default_image_features=settings['default_image_features']
default_video_features=settings['default_video_features']
default_csv_features=settings['default_csv_features']

# now assess folders by content type 
totalfiles, sampletypes, counts = classifyfolder(os.listdir())
print('-----------------------------------')
print('DETECTED %s FILES (%s)'%(str(totalfiles), str(sampletypes)))
print('-----------------------------------')
print(counts)

##########################################################
##                     FEATURIZATION                    ##
##########################################################
# go to the load_directory 
os.chdir(prevdir+'/load_dir')
load_dir=os.getcwd()

# now based on filetypes, featurize accordingly (can probably compress this into a loop)
if 'audio' in sampletypes:
    # import right featurizers (based on models)
    print('-----------------------------------')
    print('AUDIO FEATURIZING - %s'%(default_audio_features.upper()))
    print('-----------------------------------')
    os.chdir(prevdir+'/features/audio_features')
    os.system('python3 featurize.py %s'%(load_dir))

if 'text' in sampletypes:
    # import right featurizers (based on models)
    print('-----------------------------------')
    print('TEXT FEATURIZING - %s'%(default_text_features.upper()))
    print('-----------------------------------')
    os.chdir(prevdir+'/features/text_features')
    os.system('python3 featurize.py %s'%(load_dir))

if 'image' in sampletypes:
    # import right featurizers (based on models)
    print('-----------------------------------')
    print('IMAGE FEATURIZING - %s'%(default_image_features.upper()))
    print('-----------------------------------')
    os.chdir(prevdir+'/features/image_features')
    os.system('python3 featurize.py %s'%(load_dir))

if 'video' in sampletypes:
    # import right featurizers (based on models)
    print('-----------------------------------')
    print('VIDEO FEATURIZING - %s'%(default_video_features))
    print('-----------------------------------')
    os.chdir(prevdir+'/features/video_features')
    os.system('python3 featurize.py %s'%(load_dir))

if 'csv' in sampletypes:
    # import right featurizers (based on models)
    print('-----------------------------------')
    print('CSV FEATURIZING - %s'%(default_csv_features))
    print('-----------------------------------')
    os.chdir(prevdir+'/features/csv_features')
    os.system('python3 featurize.py %s'%(load_dir))

##########################################################
##                GET MODEL PREDICTIONS                 ##
##########################################################

# class_list.append(classname) --> could be label here 
os.chdir(prevdir+'/models')
listdir=os.listdir()
model_dir=os.getcwd()

if 'audio_models' in listdir:
    print('-----------------------------------')
    print('AUDIO MODELING - %s'%(default_audio_features.upper()))
    print('-----------------------------------')
    os.chdir(prevdir+'/models/audio_models')
    make_predictions('audio', default_audio_features, os.getcwd(), load_dir)
if 'text_models' in listdir:
    print('-----------------------------------')
    print('TEXT MODELING - %s'%(default_text_features.upper()))
    print('-----------------------------------')
    os.chdir(prevdir+'/models/text_models')
    make_predictions('text', default_text_features,  os.getcwd(), load_dir)
if 'image_models' in listdir:
    print('-----------------------------------')
    print('IMAGE MODELING - %s'%(default_image_features.upper()))
    print('-----------------------------------')
    os.chdir(prevdir+'/models/image_models')
    make_predictions('image', default_image_features, os.getcwd(), load_dir)
if 'video_models' in listdir:
    print('-----------------------------------')
    print('VIDEO MODELING - %s'%(default_video_features.upper()))
    print('-----------------------------------')
    os.chdir(prevdir+'/models/video_models')
    make_predictions('video', default_video_features, os.getcwd(), load_dir)
if 'csv_models' in listdir:
    print('-----------------------------------')
    print('CSV FEATURIZING - %s'%(default_csv_features.upper()))
    print('-----------------------------------')
    os.chdir(prevdir+'/models/csv_models')
    make_predictions('csv', default_csv_features, os.getcwd(), load_dir)

