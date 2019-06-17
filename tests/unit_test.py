'''
Simple unit test the default parameters 
of the repository. 

Note this unit_testing requires the settings.json
to defined in the base directory.
'''
import unittest, os, shutil 

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


cur_dir = os.getcwd()
prevdir= prev_dir(cur_dir)
load_dir = prevdir+'/load_dir'
train_dir = prevdir + '/train_dir'

class SimplisticTest(unittest.TestCase):
     
    def test(self):
        a = 'a'
        b = 'a'
        self.assertEqual(a, b)
    
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
            import protobuf
            import PyYAML
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
    def test_brew(self):
        # test brew installation by merging two test files 
        os.system('sox test_audio.wav test_audio.wav test2.wav')
        if 'test2.wav' in os.listdir():
            b=True 
            os.remove('test2.wav')  
        else:
            b=False    
        self.assertEqual(True, b)      

    # brew installation (FFmpeg)
    def test_FFmpeg(self):
        # test FFmpeg installation with test_audio file conversion 
        os.system('ffmpeg -i test_audio.wav test_audio.mp3')
        listdir=os.listdir()
        if 'test_audio.mp3' in listdir:
            b=True 
            os.remove('test_audio.mp3')
        else:
            b=False 
        self.assertEqual(True, b)
        
    ###############################################################
    ##                    TRAINING TESTS                         ##
    ###############################################################

    # can train machine learning model via specified trainer (audio, text, image, video, and .CSV files)

    # can compress models (SC vs. deep learning) 

    ###############################################################
    ##            FEATURIZATION / LOADING TESTS                  ##
    ###############################################################

    # can featurize audio files via specified featurizer (can be all featurizers) 
    # can load model via specified training script 

    shutil.copy(os.getcwd()+'/test_audio.wav', load_dir+'/test_audio.wav')
    shutil.copy(os.getcwd()+'/test_text.txt', load_dir+'/test_text.txt')
    shutil.copy(os.getcwd()+'/test_image.png', load_dir+'/test_image.png')
    shutil.copy(os.getcwd()+'/test_video.mp4', load_dir+'/test_video.mp4')
    shutil.copy(os.getcwd()+'/test_csv.csv', load_dir+'/test_csv.csv')
     
if __name__ == '__main__':
    unittest.main()
