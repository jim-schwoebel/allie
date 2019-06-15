'''
fingerprint audio models in a streaming folder. 

first determine the file types available in the folder + load appropriate featurizers, as necessary.

change to load directory 
'''
import os

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
    values=counts.values()
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
            os.rename(listdir[i], listdir[i][0:-4]+'_audio'+listdir[-4:])
        elif listdir[i].endswith(('.png', '.jpg')):
            os.rename(listdir[i], listdir[i][0:-4]+'_image'+listdir[-4:])
        elif listdir[i].endswith(('.txt')):
            listdir[i], listdir[i][0:-4]+'_text'+listdir[-4:]
        elif listdir[i].endswith(('.mp4', '.avi')):
            listdir[i], listdir[i][0:-4]+'_video'+listdir[-4:]
        elif listdir[i].endswith(('.csv')):
            listdir[i], listdir[i][0:-4]+'_csv'+listdir[-4:]

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

    return data 

def load_tpot():
    # load the TPOT file 
    # make a prediction 

def load_hypsklearn():
    # load the hypsklearn pickle file 
    # make a prediction 

def load_scsr():
    # load the pickle file (or joblib if compressed)
    # make a prediction 

def load_devol():
     # load in the model file (compressed or not)

def load_keras():
    # load in the model file (compressed or not)

def load_ludwig():
    # make a .CSV output of all the features in the folder 
    # make a ludwig prediction
    # load that output prediction 

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

os.chdir(prevdir+'/models/audio_models')
data=detect_models()

os.chdir(prevdir+'/models/text_models')
data=detect_models()

os.chdir(prevdir+'/models/image_models')
data=detect_models()

os.chdir(prevdir+'/models/video_models')
data=detect_models()

os.chdir(prevdir+'/models/csv_models')
data=detect_models()

##########################################################
##                     FEATURIZATION                    ##
##########################################################

# now based on filetypes, featurize accordingly (can probably compress this into a loop)
if 'audio' in sampletypes:
    # import right featurizers (based on models)
    os.chdir(prevdir+'/features/audio_features')
    os.system('python3 featurize.py %s'%(os.getcwd()))

if 'text': in sampletypes:
    # import right featurizers (based on models)
    os.chdir(prevdir+'/features/text_features')
    os.system('python3 featurize.py %s'%(os.getcwd()))

if 'image' in sampletypes:
    # import right featurizers (based on models)
    os.chdir(prevdir+'/features/image_features')
    os.system('python3 featurize.py %s'%(os.getcwd()))

if 'video' in sampletypes:
    # import right featurizers (based on models)
    os.chdir(prevdir+'/features/video_features')
    os.system('python3 featurize.py %s'%(os.getcwd()))

if 'csv' in sampletypes:
    # import right featurizers (based on models)
    os.chdir(prevdir+'/features/csv_features')
    os.system('python3 featurize.py %s'%(os.getcwd()))


##########################################################
##                GET MODEL PREDICTIONS                 ##
##########################################################

# --> output .