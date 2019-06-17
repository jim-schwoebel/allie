'''
Simple unit test the default parameters 
of the repository. 

Note this unit_testing requires the settings.json
to defined in the base directory.
'''
import unittest, os, shutil, time 

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

def seed_files(filename, cur_dir, train_dir):
    os.chdir(train_dir)
    for i in range(20):
        shutil.copy(cur_dir+'/'+filename, train_dir+'/'+filename)
        os.rename(filename, str(i)+'_'+filename[-4:])

def find_model(b):
    listdir=os.listdir()
    for i in range(len(listdir)):
        if listdir[i].find('one_two') > 0 and listdir[i].endswith('.h5'):
            b=True
            break 
        elif listdir[i].find('one_two') > 0 and listdir[i].endswith('.pickle'):
            b=True
            break 
        elif listdir[i].find('one_two') > 0 and listdir[i].find('.') < 0:
            b = True 
            break 

    return b 

###############################################################
##                    INITIALIZATION.                        ##
###############################################################

cur_dir = os.getcwd()
prevdir= prev_dir(cur_dir)
load_dir = prevdir+'/load_dir'
train_dir = prevdir + '/train_dir'
model_dir = prevdir+ '/training'
loadmodel_dir = prevdir+'/models'

###############################################################
##                        UNIT TESTS                          ##
###############################################################

class SimplisticTest(unittest.TestCase):
     
    ###############################################################
    ##                       MODULE TESTS                        ##
    ###############################################################
    # confirm that all the modules are installed correctly 
    def test_requirements(self):
        # import all from requirements.txt 
        try:
            import audioread
            import decorator
            import enum 
            import h5py
            import html5lib
            import joblib
            import keras
            import librosa
            import llvmlite
            import numba
            import numpy
            import yaml
            import resampy
            import sklearn 
            import scipy
            import soundfile 
            import tensorflow
            import cv2 
            import webrtcvad
            import xgboost
            import tpot
            import beautifultable
            import alphapy
            import hyperopt 
            import tqdm
            b=True
        except:
           b=False 

        self.assertEqual(True, b)  
    
    # brew installations (SoX)
    def test_sox(self):
        # test brew installation by merging two test files 
        os.system('sox test_audio.wav test_audio.wav test2.wav')
        if 'test2.wav' in os.listdir():
            b=True  
        else:
            b=False    
        self.assertEqual(True, b)      

    # brew installation (FFmpeg)
    def test_ffmpeg(self):
        # test FFmpeg installation with test_audio file conversion 
        os.system('ffmpeg -i test_audio.wav test_audio.mp3')
        if 'test_audio.mp3' in os.listdir():
            b=True 
        else:
            b=False 
        self.assertEqual(True, b)
        
    ###############################################################
    ##                    TRAINING TESTS                         ##
    ###############################################################

    

    # test audio file training 
    os.chdir(train_dir)
    os.mkdir('one')
    os.mkdir('two')
    seed_files('test_audio.wav', cur_dir, train_dir+'/one')
    seed_files('test_audio.wav', cur_dir, train_dir+'/two')
    os.chdir(model_dir)
    os.system('python3 model.py audio 2 c one two')
    os.chdir(train_dir)
    shutil.rmtree('one')
    shutil.rmtree('two')

    # text file training
    os.chdir(train_dir)
    os.mkdir('one')
    os.mkdir('two')
    seed_files('test_text.txt', cur_dir, train_dir+'/one')
    seed_files('test_text.txt', cur_dir, train_dir+'/two')
    os.chdir(model_dir)
    os.system('python3 model.py text 2 c one two')
    os.chdir(train_dir)
    shutil.rmtree('one')
    shutil.rmtree('two')

    # image file training
    os.chdir(train_dir)
    os.mkdir('one')
    os.mkdir('two')
    seed_files('test_image.png', cur_dir, train_dir+'/one')
    seed_files('test_image.png', cur_dir, train_dir+'/two')
    os.chdir(model_dir)
    os.system('python3 model.py image 2 c one two')
    os.chdir(train_dir)
    shutil.rmtree('one')
    shutil.rmtree('two')

    # video file training
    os.chdir(train_dir)
    os.mkdir('one')
    os.mkdir('two')
    seed_files('test_video.mp4', cur_dir, train_dir+'/one')
    seed_files('test_video.mp4', cur_dir, train_dir+'/two')
    os.chdir(model_dir)
    os.system('python3 model.py video 2 c one two')
    os.chdir(train_dir)
    shutil.rmtree('one')
    shutil.rmtree('two')

    # csv file training
    os.chdir(train_dir)
    os.mkdir('one')
    os.mkdir('two')
    seed_files('test_csv.csv', cur_dir, train_dir+'/one')
    seed_files('test_csv.csv', cur_dir, train_dir+'/two')
    os.chdir(model_dir)
    os.system('python3 model.py csv 2 c one two')
    os.chdir(train_dir)
    shutil.rmtree('one')
    shutil.rmtree('two')

    # now do all the tests 
    os.chdir(loadmodel_dir+'/audio_models')

    def test_audiomodel(self):
        b=False
        b = find_model(b)
        self.assertEqual(True, b)

    os.chdir(loadmodel_dir+'/text_models')

    def test_textmodel(self):
        b=False
        b = find_model(b)
        self.assertEqual(True, b)

    os.chdir(loadmodel_dir+'/image_models')

    def test_imagemodel(self):
        b=False
        b = find_model(b)
        self.assertEqual(True, b)

    os.chdir(loadmodel_dir+'/video_models')

    def video_audiomodel(self):
        b=False
        b = find_model(b)
        self.assertEqual(True, b)

    os.chdir(loadmodel_dir+'/csv_models')

    def test_csvmodel(self):
        b=False
        b = find_model(b)
        self.assertEqual(True, b)

    ###############################################################
    ##            FEATURIZATION / LOADING TESTS                  ##
    ###############################################################

    # can featurize audio files via specified featurizer (can be all featurizers) 
    # can also load recently trained models 

    shutil.copy(cur_dir+'/test_audio.wav', load_dir+'/test_audio.wav')
    shutil.copy(cur_dir+'/test_text.txt', load_dir+'/test_text.txt')
    shutil.copy(cur_dir+'/test_image.png', load_dir+'/test_image.png')
    shutil.copy(cur_dir+'/test_video.mp4', load_dir+'/test_video.mp4')
    shutil.copy(cur_dir+'/test_csv.csv', load_dir+'/test_csv.csv')

    os.chdir(loadmodel_dir)
    os.system('python3 load_models.py')

    os.chdir(load_dir)

    # note we have to do a loop here to end where the end is 
    # 'audio.json' | 'text.json' | 'image.json' | 'video.json' | 'csv.json'
    # this is because the files are renamed to not have conflicts.
    # for example, if 'audio.wav' --> 'audio.json' and 'audio.mp4' --> 'audio.json',
    # both would have a conflicting name and would overwrite each other. 

    def test_loadaudio(self, load_dir=load_dir):
        os.chdir(load_dir)
        listdir=os.listdir() 
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('audio.json') > 0: 
                b=True
                break
        self.assertEqual(True, b)

    def test_loadtext(self, load_dir=load_dir):
        os.chdir(load_dir)
        listdir=os.listdir()
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('text.json') > 0: 
                b=True
                break
        self.assertEqual(True, b)

    def test_loadimage(self, load_dir=load_dir):
        os.chdir(load_dir)
        listdir=os.listdir()
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('image.json') > 0: 
                b=True
                break
        self.assertEqual(True, b)

    def test_loadvideo(self, load_dir=load_dir):
        os.chdir(load_dir)
        listdir=os.listdir()
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('video.json') > 0: 
                b=True
                break
        self.assertEqual(True, b)

    def test_loadcsv(self, load_dir=load_dir):
        os.chdir(load_dir)
        listdir=os.listdir()
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('csv.json') > 0: 
                b=True
                break
        self.assertEqual(True, b)

if __name__ == '__main__':
    unittest.main()

