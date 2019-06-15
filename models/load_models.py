'''
fingerprint audio models in a streaming folder. 

first determine the file types available in the folder + load appropriate featurizers, as necessary.

change to load directory 
'''
import os, json 

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
    # this script renames all the files in the current directory to avoid conflicts 
    listdir=os.listdir() 
    for i in range(len(listdir)):
        if listdir[i].endswith(('.mp3', '.wav')):
            os.rename(listdir[i], listdir[i][0:-4]+'_audio'+listdir[i][-4:])
        elif listdir[i].endswith(('.png', '.jpg')):
            os.rename(listdir[i], listdir[i][0:-4]+'_image'+listdir[i][-4:])
        elif listdir[i].endswith(('.txt')):
            os.rename(listdir[i], listdir[i][0:-4]+'_text'+listdir[i][-4:])
        elif listdir[i].endswith(('.mp4', '.avi')):
            os.rename(listdir[i], listdir[i][0:-4]+'_video'+listdir[i][-4:])
        elif listdir[i].endswith(('.csv')):
            os.rename(listdir[i], listdir[i][0:-4]+'_csv'+listdir[i][-4:])

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
        if listdir[i].find('TPOT.pickle'):
            tpot_models.append(listdir[i])
        elif listdir[i].find('scsr.pickle'):
            scsr_models.append(listdir[i])
        elif listdir[i].find('hypsklearn.pickle'):
            hypsklearn_models.append(listdir[i])
        elif listdir[i].find('keras.h5'):
            keras_models.append(listdir[i])
        elif listdir[i].find('ludwig'):
            ludwig_models.append(listdir[i])

    # make a dictionary outputting all models
    data={'tpot_models': tpot_models,
          'scsr_models': scsr_models,
          'devol_models': devol_models,
          'keras_models': keras_models,
          'ludwig_models': ludwig_models,
         }
    print(data)

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
default_csv_features='csv'

# now assess folders by content type 
totalfiles, sampletypes, counts = classifyfolder(os.listdir())
print('-----------------------------------')
print('DETECTED %s FILES (%s)'%(str(totalfiles), str(sampletypes)))
print('-----------------------------------')
print(counts)

##########################################################
##                LOAD MODEL INFORMATION                ##
##########################################################

# for i in range(len(listdir)):
    # if listdir[i][-7:]=='.pickle':
        # model_list.append(listdir[i])

# g=json.load(open(modelname[0:-7]+'.json'))
# model_acc.append(g['accuracy'])
# deviations.append(g['deviation'])
# modeltypes.append(g['modeltype'])

# class_list.append(classname) --> could be label here 
os.chdir(prevdir+'/models')
listdir=os.listdir()

if 'audio_models' in listdir:
    os.chdir(prevdir+'/models/audio_models')
    data=detect_models()
if 'text_models' in listdir:
    os.chdir(prevdir+'/models/text_models')
    data=detect_models()
if 'image_models' in listdir:
    os.chdir(prevdir+'/models/image_models')
    data=detect_models()
if 'video_models' in listdir:
    os.chdir(prevdir+'/models/video_models')
    data=detect_models()
if 'csv_models' in listdir:
    os.chdir(prevdir+'/models/csv_models')
    data=detect_models()

##########################################################
##                     FEATURIZATION                    ##
##########################################################
# go to the load_directory 
os.chdir(prevdir+'/load_dir')
load_dir=os.getcwd()

# now based on filetypes, featurize accordingly (can probably compress this into a loop)
if 'audio' in sampletypes:
    # import right featurizers (based on models)
    os.chdir(prevdir+'/features/audio_features')
    os.system('python3 featurize.py %s'%(load_dir))

if 'text' in sampletypes:
    # import right featurizers (based on models)
    os.chdir(prevdir+'/features/text_features')
    os.system('python3 featurize.py %s'%(load_dir))

if 'image' in sampletypes:
    # import right featurizers (based on models)
    os.chdir(prevdir+'/features/image_features')
    os.system('python3 featurize.py %s'%(load_dir))

if 'video' in sampletypes:
    # import right featurizers (based on models)
    os.chdir(prevdir+'/features/video_features')
    os.system('python3 featurize.py %s'%(load_dir))

if 'csv' in sampletypes:
    # import right featurizers (based on models)
    os.chdir(prevdir+'/features/csv_features')
    os.system('python3 featurize.py %s'%(load_dir))


##########################################################
##                GET MODEL PREDICTIONS                 ##
##########################################################

# --> output .