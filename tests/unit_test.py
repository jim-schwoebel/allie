'''
Simple unit test the default parameters 
of the repository. 

Note this unit_testing requires the settings.json
to defined in the base directory.
'''
import unittest, os, shutil, time, uuid, random, json, markovify 
import pandas as pd 

###############################################################
##                  HELPER FUNCTIONS                         ##
###############################################################

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

def seed_files(filename, cur_dir, train_dir, clean_data, augment_data, filetype):
    os.chdir(train_dir)

    def text_record(filename, text_model):
        textfile=open(filename, 'w')
        # Print five randomly-generated sentences
        for i in range(5):
            textfile.write(text_model.make_sentence())
        textfile.close()

    # Get raw text as string (the Brother's Karamazov, one of my fav novels)
    with open(cur_dir+'/helpers/text.txt') as f:
        text = f.read()
    # Build the model.
    text_model = markovify.Text(text)
    for i in range(20):
        filename=str(uuid.uuid4())+'.txt'
        text_record(filename, text_model)

def find_model(b):
    listdir=os.listdir()
    b=0
    for i in range(len(listdir)):
        if listdir[i].find('one_two') == 0 and listdir[i].endswith('.h5'):
            b=b+1 
        elif listdir[i].find('one_two') == 0 and listdir[i].endswith('.pickle'):
            b=b+1
        elif listdir[i].find('one_two') == 0 and listdir[i].endswith('.joblib'):
            b=b+1
        elif listdir[i].find('one_two') > 0 and listdir[i].find('.') < 0:
            b = b+1

    # 6=non-compressed, 12=compressed models 
    if b == 6*2:
        b=True
    else:
        b=False 

    print(b)

    return b

def clean_file(directory, clean_dir, train_dir):
    os.chdir(clean_dir)
    os.system('python3 clean.py %s %s'%(clean_dir, train_dir+'/'+directory))

def augment_file(directory, augment_dir, train_dir):
    os.chdir(augment_dir)
    os.system('python3 augment.py %s'%(train_dir+'/'+directory))

###############################################################
##                     INITIALIZATION                        ##
###############################################################

# initialize variables for the test 
cur_dir=os.getcwd()
prevdir= prev_dir(cur_dir)
load_dir = prevdir+'/load_dir'
train_dir = prevdir + '/train_dir'
model_dir = prevdir+ '/training'
features_dir=prevdir+'/features'
loadmodel_dir = prevdir+'/models'
production_dir=prevdir+'/production'
clean_dir=prevdir+'/datasets/cleaning/'
augment_dir=prevdir+'/datasets/augmentation'

# settings
settings=json.load(open(prevdir+'/settings.json'))
clean_data=settings['clean_data']
augment_data=settings['augment_data']

# transcript settings 
default_audio_transcript=settings['default_audio_transcriber']
default_image_transcript=settings['default_image_transcriber']
default_text_transcript=settings['default_text_transcriber']
default_video_transcript=settings['default_video_transcriber']
default_csv_transcript=settings['default_csv_transcriber']
# feature settings 
default_audio_features=settings['default_audio_features']
default_text_features=settings['default_text_features']
default_image_features=settings['default_image_features']
default_video_features=settings['default_video_features']
default_csv_features=settings['default_csv_features']

# other settings for raining scripts 
training_scripts=settings['default_training_script']
production=settings['create_YAML']
model_compress=settings['model_compress']

# directories 
audiodir=loadmodel_dir+'/audio_models'
textdir=loadmodel_dir+'/text_models'
imagedir=loadmodel_dir+'/image_models'
videodir=loadmodel_dir+'/video_models'
csvdir=loadmodel_dir+'/csv_models'

###############################################################
##                        UNIT TESTS                         ##
###############################################################

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_dependencies(unittest.TestCase):
    '''
    confirm that all the modules are installed correctly, along with
    all brew install commands. 
    '''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

    def test_requirements(self):
        # import all from requirements.txt 
        try:
            import alphapy, audioread, beautifultable, decorator
            from enum import Enum
            import eyed3, ffmpeg_normalize, gensim
            import h5py, hyperopt, joblib, kafka, keras, librosa, llvmlite, ludwig
            import matplotlib, nltk, numba, numpy, cv2, moviepy, peakutils, PIL
            import pocketsphinx, parselmouth, pydub, pymongo, python_speech_features
            # ignore protobuf
            import pyscreenshot, pytesseract, yaml, requests, resampy, sklearn 
            import scipy, skvideo, simplejson, sounddevice, soundfile, spacy, speech_recognition
            import tensorflow, textblob, tpot, tqdm, wave, webrtcvad, wget, xgboost
            b=True
        except:
            b=False 

        self.assertEqual(True, b)  
    
    # brew installations (SoX)
    def test_sox(self):

        # test brew installation by merging two test files 
        os.chdir(cur_dir)
        os.system('sox test_audio.wav test_audio.wav test2.wav')
        if 'test2.wav' in os.listdir():
            b=True  
        else:
            b=False    
        self.assertEqual(True, b)      

    # brew installation (FFmpeg)
    def test_c_ffmpeg(self):

        # test FFmpeg installation with test_audio file conversion 
        os.chdir(cur_dir)
        os.system('ffmpeg -i test_audio.wav test_audio.mp3')
        if 'test_audio.mp3' in os.listdir():
            b=True 
        else:
            b=False 
        self.assertEqual(True, b)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_cleaning(unittest.TestCase):
    '''
    tests file cleaning capabilities by removing duplicates, etc.
    across all file types.
    '''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

    # audio file cleaning
    def test_audio_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
        directory='audio_clean'
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        os.chdir(directory)
        for i in range(20):
            shutil.copy(cur_dir+'/test_audio.wav', train_dir+'/'+directory+'/test_audio.wav')
            os.rename('test_audio.wav', str(i)+'_test_audio.wav')

        clean_file(directory, clean_dir, train_dir)
        os.chdir(train_dir+'/'+directory)
        listdir=os.listdir()
        b=False 
        if len(listdir) == 1:
            b=True
        self.assertEqual(True, b) 
        # remove temp directory 
        os.chdir(train_dir)
        shutil.rmtree(train_dir+'/'+directory)

    # text file cleaning 
    def test_text_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
        directory='text_clean'
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        os.chdir(directory)
        for i in range(20):
            shutil.copy(cur_dir+'/test_audio.wav', train_dir+'/'+directory+'/test_audio.wav')
            os.rename('test_text.txt', str(i)+'_test_text.txt')

        clean_file(directory, clean_dir, train_dir)
        os.chdir(train_dir+'/'+directory)
        listdir=os.listdir()
        b=False 
        if len(listdir) == 1:
            b=True
        self.assertEqual(True, b) 
        # remove temp directory 
        os.chdir(train_dir)
        shutil.rmtree(train_dir+'/'+directory)

    # image file cleaning 
    def test_image_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
        directory='image_clean'
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        os.chdir(directory)
        for i in range(20):
            shutil.copy(cur_dir+'/test_image.png', train_dir+'/'+directory+'/test_image.png')
            os.rename('test_image.png', str(i)+'_test_iamge.png')

        clean_file(directory, clean_dir, train_dir)
        os.chdir(train_dir+'/'+directory)
        listdir=os.listdir()
        b=False 
        if len(listdir) == 1:
            b=True
        self.assertEqual(True, b) 
        # remove temp directory 
        os.chdir(train_dir)
        shutil.rmtree(train_dir+'/'+directory)

    # video file cleaning
    def test_video_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
        directory='video_clean'
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        os.chdir(directory)
        for i in range(20):
            shutil.copy(cur_dir+'/test_video.mp4', train_dir+'/'+directory+'/test_video.mp4')
            os.rename('test_video.mp4', str(i)+'_test_video.mp4')

        clean_file(directory, clean_dir, train_dir)
        os.chdir(train_dir+'/'+directory)
        listdir=os.listdir()
        b=False 
        if len(listdir) == 1:
            b=True
        self.assertEqual(True, b) 
        # remove temp directory 
        os.chdir(train_dir)
        shutil.rmtree(train_dir+'/'+directory)

    # csv file cleaning
    def test_csv_clean(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
        directory='csv_clean'
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        os.chdir(directory)
        for i in range(20):
            shutil.copy(cur_dir+'/test_csv.csv', train_dir+'/'+directory+'/test_csv.csv')
            os.rename('test_csv.csv', str(i)+'_test_csv.csv')

        clean_file(directory, clean_dir, train_dir)
        os.chdir(train_dir+'/'+directory)
        listdir=os.listdir()
        b=False 
        if len(listdir) == 1:
            b=True
        self.assertEqual(True, b) 
        # remove temp directory 
        os.chdir(train_dir)
        shutil.rmtree(train_dir+'/'+directory)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_augmentation(unittest.TestCase):
    '''
    tests augmentation capabilities for all data types.
    '''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

    # audio features
    def test_audio_augment(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
        directory='audio_augment'
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        shutil.copy(cur_dir+'/test_audio.wav', train_dir+'/'+directory+'/test_audio.wav')
        augment_file(directory, augment_dir, train_dir)
        os.chdir(train_dir+'/'+directory)
        listdir=os.listdir()
        b=False 
        if len(listdir) == 13:
            b=True
        self.assertEqual(True, b) 
        # remove temp directory 
        os.chdir(train_dir)
        shutil.rmtree(train_dir+'/'+directory)
        
    # text features
    def test_text_augment(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
        directory='text_augment'
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)
        shutil.copy(cur_dir+'/test_text.txt', train_dir+'/'+directory+'/test_text.txt')
        augment_file(directory, augment_dir, train_dir)
        os.chdir(train_dir+'/'+directory)
        listdir=os.listdir()
        b=False 
        if len(listdir) == 1:
            b=True
        self.assertEqual(True, b) 
        # remove temp directory 
        os.chdir(train_dir)
        shutil.rmtree(train_dir+'/'+directory)

    # image features
    def test_image_augment(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
        directory='image_augment'
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)
        shutil.copy(cur_dir+'/test_image.png', train_dir+'/'+directory+'/test_image.png')
        augment_file(directory, augment_dir, train_dir)
        os.chdir(train_dir+'/'+directory)
        listdir=os.listdir()
        b=False 
        if len(listdir) == 1:
            b=True
        self.assertEqual(True, b) 
        # remove temp directory 
        os.chdir(train_dir)
        shutil.rmtree(train_dir+'/'+directory)

    # video features
    def test_video_augment(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
        directory='video_augment'
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)
        shutil.copy(cur_dir+'/test_video.mp4', train_dir+'/'+directory+'/test_video.mp4')
        augment_file(directory, augment_dir, train_dir)
        os.chdir(train_dir+'/'+directory)
        listdir=os.listdir()
        b=False 
        if len(listdir) == 1:
            b=True
        self.assertEqual(True, b) 
        # remove temp directory 
        os.chdir(train_dir)
        shutil.rmtree(train_dir+'/'+directory)

    # csv features 
    def test_csv_augment(self, clean_dir=clean_dir, train_dir=train_dir, cur_dir=cur_dir, clean_data=clean_data, augment_data=augment_data):
        directory='csv_augment'
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)
        shutil.copy(cur_dir+'/test_csv.csv', train_dir+'/'+directory+'/test_csv.csv')
        augment_file(directory, augment_dir, train_dir)
        os.chdir(train_dir+'/'+directory)
        listdir=os.listdir()
        b=False 
        if len(listdir) == 1:
            b=True
        self.assertEqual(True, b) 
        # remove temp directory 
        os.chdir(train_dir)
        shutil.rmtree(train_dir+'/'+directory)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_features(unittest.TestCase):
    '''
    tests featurization capabilities across all training scripts.
    '''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

    # change settings.json to include every type of featurization
    # change back to default settings.

    # audio features
    def test_audio_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir):
        directory='audio_features'
        folder=train_dir+'/'+directory
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        # put test audio file in directory 
        shutil.copy(cur_dir+'/test_audio.wav', folder+'/test_audio.wav')

        os.chdir(features_dir+'/audio_features/')
        features_list=["audioset_features", "audiotext_features", "librosa_features", 
                        "meta_features", "mixed_features", "myprosody_features", 
                        "praat_features", "pspeech_features", "pyaudio_features", 
                        "sa_features", "sox_features", "specimage_features", 
                        "specimage2_features", "spectrogram_features", "standard_features"]

        # features that don't work
        # ['audioset_features', 'mixed_features', 'myprosody_features']

        for i in range(len(features_list)):
            print('------------------------------')
            print('FEATURIZING - %s'%(features_list[i].upper()))
            print('------------------------------')
            os.system('python3 featurize.py %s %s'%(folder, features_list[i]))

        # now that we have the folder let's check if the array has all the features
        os.chdir(folder)
        g=json.load(open('test_audio.json'))
        features=g['features']['audio']
        print(list(features))
        if list(features) == features_list:
            b=True
        else:
            b=False 

        self.assertEqual(True, b) 
        os.chdir(train_dir)
        shutil.rmtree(directory)
        
    # text features
    def test_text_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir):
        directory='text_features'
        folder=train_dir+'/'+directory
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        # put test audio file in directory 
        shutil.copy(cur_dir+'/test_text.txt', folder+'/test_text.txt')

        os.chdir(features_dir+'/text_features/')
        features_list=["fast_features", "glove_features", "nltk_features", 
                       "spacy_features", "w2vec_features"]

        # features that don't work
        # features = ['w2vec_features']

        for i in range(len(features_list)):
            print('------------------------------')
            print('FEATURIZING - %s'%(features_list[i].upper()))
            print('------------------------------')
            os.system('python3 featurize.py %s %s'%(folder, features_list[i]))

        # now that we have the folder let's check if the array has all the features
        os.chdir(folder)
        g=json.load(open('test_text.json'))
        features=g['features']['text']
        if list(features) == features_list:
            b=True
        else:
            b=False 

        self.assertEqual(True, b) 
        os.chdir(train_dir)
        shutil.rmtree(directory)

    # image features
    def test_image_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir):
        directory='image_features'
        folder=train_dir+'/'+directory
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        # put test audio file in directory 
        shutil.copy(cur_dir+'/test_image.png', folder+'/test_image.png')

        os.chdir(features_dir+'/image_features/')
        features_list=["image_features", "inception_features", "resnet_features", 
                       "tesseract_features", "vgg16_features", "vgg19_features", "xception_features"]

        # features that dont work 
        # features=['VGG19 features']

        for i in range(len(features_list)):
            print('------------------------------')
            print('FEATURIZING - %s'%(features_list[i].upper()))
            print('------------------------------')
            os.system('python3 featurize.py %s %s'%(folder, features_list[i]))

        # now that we have the folder let's check if the array has all the features
        os.chdir(folder)
        g=json.load(open('test_image.json'))
        features=g['features']['image']

        # features that don't work 
        # features = []
        if list(features) == features_list:
            b=True
        else:
            b=False 

        self.assertEqual(True, b) 
        os.chdir(train_dir)
        shutil.rmtree(directory)

    # video features
    def test_video_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir):
        directory='video_features'
        folder=train_dir+'/'+directory
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        # put test audio file in directory 
        shutil.copy(cur_dir+'/test_video.mp4', folder+'/test_video.mp4')

        os.chdir(features_dir+'/video_features/')
        features_list=["video_features", "y8m_features"]

        # features that don't work 
        # features = ['y8m_features']

        for i in range(len(features_list)):
            print('------------------------------')
            print('FEATURIZING - %s'%(features_list[i].upper()))
            print('------------------------------')
            os.system('python3 featurize.py %s %s'%(folder, features_list[i]))

        # now that we have the folder let's check if the array has all the features
        os.chdir(folder)
        g=json.load(open('test_video.json'))
        features=g['features']['video']
        if list(features) == features_list:
            b=True
        else:
            b=False 

        self.assertEqual(True, b) 
        os.chdir(train_dir)
        shutil.rmtree(directory)

    # csv features 
    def test_csv_features(self, features_dir=features_dir, train_dir=train_dir, cur_dir=cur_dir):
        directory='csv_features'
        folder=train_dir+'/'+directory
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        # put test audio file in directory 
        shutil.copy(cur_dir+'/test_csv.csv', folder+'/test_csv.csv')

        os.chdir(features_dir+'/csv_features/')
        features_list=  ["csv_features"]

        for i in range(len(features_list)):
            print('------------------------------')
            print('FEATURIZING - %s'%(features_list[i].upper()))
            print('------------------------------')
            os.system('python3 featurize.py %s %s'%(folder, features_list[i]))

        # now that we have the folder let's check if the array has all the features
        os.chdir(folder)
        g=json.load(open('test_csv.json'))
        features=g['features']['csv']
        if list(features) == features_list:
            b=True
        else:
            b=False 

        self.assertEqual(True, b) 
        os.chdir(train_dir)
        shutil.rmtree(directory)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_transcription(unittest.TestCase):
    '''
    tests the ability to transcribe across many
    data types
    '''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

    # audio transcription
    def test_audio_transcription(self, train_dir=train_dir, cur_dir=cur_dir):
        os.chdir(train_dir)
        directory='audio_transcription'
        folder=train_dir+'/'+directory
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        # put test audio file in directory 
        shutil.copy(cur_dir+'/test_audio.wav', folder+'/test_audio.wav')

        os.chdir(features_dir+'/audio_features/')
        os.system('python3 featurize.py %s'%(folder))

        # now that we have the folder let's check if the array has all the features
        os.chdir(folder)
        g=json.load(open('test_audio.json'))
        transcripts=list(g['transcripts']['audio'])

        if default_audio_transcript in transcripts:
            b=True
        else:
            b=False 

        self.assertEqual(True, b) 
        os.chdir(train_dir)
        shutil.rmtree(directory)

    # text transcription
    def test_text_transcription(self, train_dir=train_dir, cur_dir=cur_dir):
        os.chdir(train_dir)
        directory='text_transcription'
        folder=train_dir+'/'+directory
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        # put test audio file in directory 
        shutil.copy(cur_dir+'/test_text.txt', folder+'/test_text.txt')

        os.chdir(features_dir+'/text_features/')
        os.system('python3 featurize.py %s'%(folder))

        # now that we have the folder let's check if the array has all the features
        os.chdir(folder)
        g=json.load(open('test_text.json'))
        transcripts=list(g['transcripts']['text'])

        if default_text_transcript in transcripts:
            b=True
        else:
            b=False 

        self.assertEqual(True, b) 
        os.chdir(train_dir)
        shutil.rmtree(directory)

    # image transcription
    def test_image_transcription(self, train_dir=train_dir, cur_dir=cur_dir):
        os.chdir(train_dir)
        directory='image_transcription'
        folder=train_dir+'/'+directory
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        # put test audio file in directory 
        shutil.copy(cur_dir+'/test_image.png', folder+'/test_image.png')

        os.chdir(features_dir+'/image_features/')
        os.system('python3 featurize.py %s'%(folder))

        # now that we have the folder let's check if the array has all the features
        os.chdir(folder)
        g=json.load(open('test_image.json'))
        transcripts=list(g['transcripts']['image'])

        if default_image_transcript in transcripts:
            b=True
        else:
            b=False 

        self.assertEqual(True, b) 
        os.chdir(train_dir)
        shutil.rmtree(directory)

    # video transcription
    def test_video_transcription(self, train_dir=train_dir, cur_dir=cur_dir):
        os.chdir(train_dir)
        directory='video_transcription'
        folder=train_dir+'/'+directory
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        # put test audio file in directory 
        shutil.copy(cur_dir+'/test_video.mp4', folder+'/test_video.mp4')

        os.chdir(features_dir+'/video_features/')
        os.system('python3 featurize.py %s'%(folder))

        # now that we have the folder let's check if the array has all the features
        os.chdir(folder)
        g=json.load(open('test_video.json'))
        transcripts=list(g['transcripts']['video'])

        if default_video_transcript in transcripts:
            b=True
        else:
            b=False 

        self.assertEqual(True, b) 
        os.chdir(train_dir)
        shutil.rmtree(directory)

    # csv transcription
    def test_csv_transcription(self, train_dir=train_dir, cur_dir=cur_dir):
        os.chdir(train_dir)
        directory='csv_transcription'
        folder=train_dir+'/'+directory
        os.chdir(train_dir)
        try:
            os.mkdir(directory)
        except:
            shutil.rmtree(directory)
            os.mkdir(directory)

        # put test audio file in directory 
        shutil.copy(cur_dir+'/test_csv.csv', folder+'/test_csv.csv')

        os.chdir(features_dir+'/csv_features/')
        os.system('python3 featurize.py %s'%(folder))

        # now that we have the folder let's check if the array has all the features
        os.chdir(folder)
        g=json.load(open('test_csv.json'))
        transcripts=list(g['transcripts']['csv'])

        if default_csv_transcript in transcripts:
            b=True
        else:
            b=False 

        self.assertEqual(True, b) 
        os.chdir(train_dir)
        shutil.rmtree(directory)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_training(unittest.TestCase):
    '''
    Tests all available training scripts 
    '''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

    def setUp(self, prevdir=prevdir):
        # change settings.json to test all model scripts 
        os.chdir(prevdir)
        settings=json.load(open('settings.json'))
        settings['default_training_script']=['scsr', 'tpot', 'hyperopt', 'keras', 'devol', 'ludwig']
        settings['clean_data']=False 
        settings['augment_data']=False 
        settings['model_compress']=True
        settings['create_YAML']=True 
        jsonfile=open('settings.json', 'w')
        json.dump(settings, jsonfile)
        jsonfile.close()

    def tearDown(self, prevdir=prevdir, training_scripts=training_scripts, clean_data=clean_data, augment_data=augment_data, model_compress=model_compress, production=production):
        # change settings.json back to normal to defaults  
        os.chdir(prevdir)
        settings=json.load(open('settings.json'))
        settings['default_training_script']=training_scripts
        settings['clean_data']=clean_data
        settings['augment_data']=augment_data 
        settings['model_compress']=model_compress
        settings['create_YAML']=production 
        jsonfile=open('settings.json','w')
        json.dump(settings, jsonfile)
        jsonfile.close() 

    def test_training(self, audiodir=textdir, cur_dir=cur_dir, train_dir=train_dir, model_dir=model_dir, clean_data=clean_data, augment_data=augment_data):

        # use text model for training arbitrarily because it's the fastest model training time.
        # note there are some risks for an infinite loop here in case model training fails 

        os.chdir(train_dir)
        try:
            os.mkdir('one')
        except:
            shutil.rmtree('one')
            os.mkdir('one')
        try:
            os.mkdir('two')
        except:
            shutil.rmtree('two')
            os.mkdir('two')
        seed_files('test_text.txt', cur_dir, train_dir+'/one', clean_data, augment_data,'text')
        seed_files('test_text.txt', cur_dir, train_dir+'/two', clean_data, augment_data,'text')
        os.chdir(model_dir)
        # iterate through all machine learning model training methods
        os.system('python3 model.py text 2 c one two')
        os.chdir(train_dir)
        shutil.rmtree('one')
        shutil.rmtree('two')
        
        # now find the model 
        os.chdir(textdir)
        b=False
        b = find_model(b)
        self.assertEqual(True, b)

        # ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
        # class test_compress(unittest.TestCase):
        #     '''
        #     tests ability to compress machine learning models
        #     '''
        # ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

        #     TEST PICKLE 
        #     load a .pickle model directory in test directory 
        #     compress this file 

        #     TEST .H5
        #     compress with keras_compressor 

        #     Delete these files 

        # ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
        # class test_production(unittest.TestCase):
        #       '''
        #       ability to create a production-ready model repository 
        #       create_YAML.py
        #       '''
        # ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

        #       Load a pickle file 
        #       Load a .H5 file for production
        #       Load a Ludwig model in production 


##### ##### ##### ##### ##### ##### ##### ##### ##### #####
class test_loading(unittest.TestCase):
    '''
    FEATURIZATION AND LOADING TESTS

    # note we have to do a loop here to end where the end is 
    # 'audio.json' | 'text.json' | 'image.json' | 'video.json' | 'csv.json'
    # this is because the files are renamed to not have conflicts.
    # for example, if 'audio.wav' --> 'audio.json' and 'audio.mp4' --> 'audio.json',
    # both would have a conflicting name and would overwrite each other. 
    '''
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

    def setUp(self, prevdir=prevdir):
        # change settings.json to test all model scripts 
        os.chdir(prevdir)
        settings=json.load(open('settings.json'))
        # set features for the right ML models 
        settings['default_audio_features']=['standard_features'] 
        settings['default_text_features']=['nltk_features'] 
        settings['default_image_features']=['image_features'] 
        settings['default_video_features']=['video_features'] 
        settings['default_csv_features']=['csv_features'] 
        jsonfile=open('settings.json', 'w')
        json.dump(settings, jsonfile)
        jsonfile.close()

    def tearDown(self, default_audio_features=default_audio_features, default_text_features=default_text_features, default_image_features=default_image_features, default_video_features=default_video_features, default_csv_features=default_csv_features):
        os.chdir(prevdir)
        settings=json.load(open('settings.json'))
        # set features back to what they were before. 
        settings['default_audio_features']=default_audio_features
        settings['default_text_features']=default_image_features
        settings['default_image_features']=default_image_features
        settings['default_video_features']=default_video_features 
        settings['default_csv_features']=default_csv_features 
        jsonfile=open('settings.json','w')
        json.dump(settings, jsonfile)
        jsonfile.close() 

    def test_loadaudio(self, load_dir=load_dir, cur_dir=cur_dir, loadmodel_dir=loadmodel_dir):
        filetype='audio'
        testfile='test_audio.wav'
        # copy machine learning model into image_model dir 
        os.chdir(cur_dir+'/helpers/models/%s_models/'%(filetype))
        listdir=os.listdir()
        temp=os.getcwd()
        tempfiles=list()

        os.chdir(loadmodel_dir)
        if '%s_models'%(filetype) not in os.listdir():
            os.mkdir('text_models')

        os.chdir(temp)
        
        for i in range(len(listdir)):
            shutil.copy(temp+'/'+listdir[i], loadmodel_dir+'/%s_models/'%(filetype)+listdir[i])
            tempfiles.append(listdir[i])

        # copy file in load_dir 
        shutil.copy(cur_dir+'/'+testfile, load_dir+'/'+testfile)
        # copy machine learning models into proper models directory 
        os.chdir(cur_dir+'/helpers/models/%s_models/'%(filetype))
        listdir=os.listdir()

        # copy audio machine learning model into directory (one_two)
        audiomodel_files=list()
        for i in range(len(listdir)):
            shutil.copy(cur_dir+'/helpers/models/%s_models/'%(filetype)+listdir[i], loadmodel_dir+'/%s_models/'%(filetype)+listdir[i])
            audiomodel_files.append(listdir[i])

        os.chdir(loadmodel_dir)
        os.system('python3 load_models.py')
        os.chdir(load_dir)

        os.chdir(load_dir)
        listdir=os.listdir() 
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('%s.json'%(filetype)) > 0: 
                b=True
                break

        # now remove all the temp files 
        os.chdir(loadmodel_dir+'/%s_models'%(filetype))
        for i in range(len(tempfiles)):
            os.remove(tempfiles[i])

        self.assertEqual(True, b)

    def test_loadtext(self, load_dir=load_dir, cur_dir=cur_dir, loadmodel_dir=loadmodel_dir):
        filetype='text'
        testfile='test_text.txt'
        # copy machine learning model into image_model dir 
        os.chdir(cur_dir+'/helpers/models/%s_models/'%(filetype))
        listdir=os.listdir()
        temp=os.getcwd()
        tempfiles=list()

        os.chdir(loadmodel_dir)
        if '%s_models'%(filetype) not in os.listdir():
            os.mkdir('text_models')

        os.chdir(temp)
        
        for i in range(len(listdir)):
            shutil.copy(temp+'/'+listdir[i], loadmodel_dir+'/%s_models/'%(filetype)+listdir[i])
            tempfiles.append(listdir[i])

        # copy file in load_dir 
        shutil.copy(cur_dir+'/'+testfile, load_dir+'/'+testfile)
        # copy machine learning models into proper models directory 
        os.chdir(cur_dir+'/helpers/models/%s_models/'%(filetype))
        listdir=os.listdir()

        # copy audio machine learning model into directory (one_two)
        audiomodel_files=list()
        for i in range(len(listdir)):
            shutil.copy(cur_dir+'/helpers/models/%s_models/'%(filetype)+listdir[i], loadmodel_dir+'/%s_models/'%(filetype)+listdir[i])
            audiomodel_files.append(listdir[i])

        os.chdir(loadmodel_dir)
        os.system('python3 load_models.py')
        os.chdir(load_dir)

        os.chdir(load_dir)
        listdir=os.listdir() 
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('%s.json'%(filetype)) > 0: 
                b=True
                break

        # now remove all the temp files 
        os.chdir(loadmodel_dir+'/%s_models'%(filetype))
        for i in range(len(tempfiles)):
            os.remove(tempfiles[i])

        self.assertEqual(True, b)

    def test_loadimage(self, load_dir=load_dir, cur_dir=cur_dir, loadmodel_dir=loadmodel_dir):
        filetype='image'
        testfile='test_image.png'
        # copy machine learning model into image_model dir 
        os.chdir(cur_dir+'/helpers/models/%s_models/'%(filetype))
        listdir=os.listdir()
        temp=os.getcwd()
        tempfiles=list()

        os.chdir(loadmodel_dir)
        if '%s_models'%(filetype) not in os.listdir():
            os.mkdir('text_models')

        os.chdir(temp)
        
        for i in range(len(listdir)):
            shutil.copy(temp+'/'+listdir[i], loadmodel_dir+'/%s_models/'%(filetype)+listdir[i])
            tempfiles.append(listdir[i])

        # copy file in load_dir 
        shutil.copy(cur_dir+'/'+testfile, load_dir+'/'+testfile)
        # copy machine learning models into proper models directory 
        os.chdir(cur_dir+'/helpers/models/%s_models/'%(filetype))
        listdir=os.listdir()

        # copy audio machine learning model into directory (one_two)
        audiomodel_files=list()
        for i in range(len(listdir)):
            shutil.copy(cur_dir+'/helpers/models/%s_models/'%(filetype)+listdir[i], loadmodel_dir+'/%s_models/'%(filetype)+listdir[i])
            audiomodel_files.append(listdir[i])

        os.chdir(loadmodel_dir)
        os.system('python3 load_models.py')
        os.chdir(load_dir)

        os.chdir(load_dir)
        listdir=os.listdir() 
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('%s.json'%(filetype)) > 0: 
                b=True
                break

        # now remove all the temp files 
        os.chdir(loadmodel_dir+'/%s_models'%(filetype))
        for i in range(len(tempfiles)):
            os.remove(tempfiles[i])

        self.assertEqual(True, b)

    def test_loadvideo(self, load_dir=load_dir, cur_dir=cur_dir, loadmodel_dir=loadmodel_dir):
        filetype='video'
        testfile='test_video.mp4'
        # copy machine learning model into image_model dir 
        os.chdir(cur_dir+'/helpers/models/%s_models/'%(filetype))
        listdir=os.listdir()
        temp=os.getcwd()
        tempfiles=list()

        os.chdir(loadmodel_dir)
        if '%s_models'%(filetype) not in os.listdir():
            os.mkdir('text_models')

        os.chdir(temp)
        
        for i in range(len(listdir)):
            shutil.copy(temp+'/'+listdir[i], loadmodel_dir+'/%s_models/'%(filetype)+listdir[i])
            tempfiles.append(listdir[i])

        # copy file in load_dir 
        shutil.copy(cur_dir+'/'+testfile, load_dir+'/'+testfile)
        # copy machine learning models into proper models directory 
        os.chdir(cur_dir+'/helpers/models/%s_models/'%(filetype))
        listdir=os.listdir()

        # copy audio machine learning model into directory (one_two)
        audiomodel_files=list()
        for i in range(len(listdir)):
            shutil.copy(cur_dir+'/helpers/models/%s_models/'%(filetype)+listdir[i], loadmodel_dir+'/%s_models/'%(filetype)+listdir[i])
            audiomodel_files.append(listdir[i])

        os.chdir(loadmodel_dir)
        os.system('python3 load_models.py')
        os.chdir(load_dir)

        os.chdir(load_dir)
        listdir=os.listdir() 
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('%s.json'%(filetype)) > 0: 
                b=True
                break

        # now remove all the temp files 
        os.chdir(loadmodel_dir+'/%s_models'%(filetype))
        for i in range(len(tempfiles)):
            os.remove(tempfiles[i])

        self.assertEqual(True, b)

if __name__ == '__main__':
    unittest.main()

