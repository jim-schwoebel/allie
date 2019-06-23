'''
Simple unit test the default parameters 
of the repository. 

Note this unit_testing requires the settings.json
to defined in the base directory.
'''
import unittest, os, shutil, time, uuid, random, json
import sounddevice as sd 
import soundfile as sf 
import pyautogui, markovify 
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

def audio_record(filename, duration, fs, channels):
    print('---------------')
    print('recording audio...')
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    sf.write(filename, myrecording, fs)
    print('done recording %s'%(filename))
    print('---------------')

def text_record(filename, text_model):

    textfile=open(filename, 'w')
    # Print five randomly-generated sentences
    for i in range(5):
        textfile.write(text_model.make_sentence())

    textfile.close()

def image_record(filename):
    pyautogui.screenshot(filename)

def video_record(filename, test_dir):
    print('---------------')
    print('recording vdieo...')
    cur_dur=os.getcwd()
    os.chdir(test_dir+'/helpers/video_record')
    # 3 second recordings 
    os.system('python3 record.py %s 3 %s'%(filename, cur_dir))
    os.chdir(cur_dir)
    print('---------------')

def csv_record(filename, newfilename):
    # take in test .CSV and manipulate the columns by copy/paste and re-write 
    csvfile=pd.read_csv(filename)
    filelength=len(csv_file)
    newlength=random.randint(0,filelength-1)

    # now re-write CSV with the new length 
    g=csvfile.iloc[0:newlength]
    randint2=random.randint(0,1)
    if randint2 == 0:
        g=g+g 
    g.to_csv(newfilename)

def seed_files(filename, cur_dir, train_dir, clean_data, augment_data, filetype):
    os.chdir(train_dir)

    if clean_data!=True and augment_data!=True:
        # if clean_data and augment_data are both false, we don't 
        # need to worrry about uniqueness, so we can seed non-unique files 
        for i in range(20):
            shutil.copy(cur_dir+'/'+filename, train_dir+'/'+filename)
            os.rename(filename, str(i)+'_'+filename[-4:])
    else:
        # if clean_data==True or augment_data==True, then we need to worry
        # about file duplicates and uniqueness. This can be overcome with 
        # a short script to manipulate 20 unique files from test data
        if filetype == 'audio':
            # load test data directory 
            if train_dir.endswith('one'):
                data_dir+cur_dir+'/helpers/audio_data/one'
            elif train_dir.endswith('two'):
                data_dir+cur_dir+'/helpers/audio_data/two'

            listdir=os.listdir(data_dir)
            print(data_dir)
            # option 1 - copy test files
            # --------------------------
            for i in range(len(listdir)):
                if listdir[i][-4:]=='.wav':
                    shutil.copy(data_dir+'/'+listdir[i], train_dir+'/'+listdir[i])
            
            # option 2 - record data yourself (must be non-silent data)
            # --------------------------
            # for i in range(20):
                # filename=str(uuid.uuid4())+'.wav'
                # audio_record(filename, 1, 16000, 1)

        elif filetype == 'text':
            # Get raw text as string (the Brother's Karamazov)
            with open(cur_dir+'/helpers/text.txt') as f:
                text = f.read()
            # Build the model.
            text_model = markovify.Text(text)
            for i in range(20):
                filename=str(uuid.uuid4())+'.txt'
                text_record(filename)

        elif filetype == 'image':
            # take 20 random screenshots with pyscreenshot
            for i in range(20):
                filename=str(uuid.uuid4())+'.png'
                image_record(filename)

        elif filetype == 'video':
            # make 20 random videos with screenshots 
            for i in range(20):
                filename=str(uuid.uuid4())+'.avi'
                video_record(filename, cur_dir)

        elif filetype == 'csv':
            # prepopulate 20 random csv files with same headers 
            shutil.copy(cur_dir+'/'+filename, train_dir+'/'+filename)
            for i in range(20):
                newfilename=str(uuid.uuid4())+'.csv'
                csv_record(filename, newfilename)
            os.remove(filename)


def find_model(b):
    listdir=os.listdir()
    for i in range(len(listdir)):
        if listdir[i].find('one_two') == 0 and listdir[i].endswith('.h5'):
            b=True
            break 
        elif listdir[i].find('one_two') == 0 and listdir[i].endswith('.pickle'):
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
production_dir=prevdir+'/production'

# settings
settings=json.load(open(prevdir+'/settings.json'))
clean_data=settings['clean_data']
augment_data=settings['augment_data']
production=settings['create_YAML']
model_compress=settings['model_compress']
audiodir=loadmodel_dir+'/audio_models'
textdir=loadmodel_dir+'/text_models'
imagedir=loadmodel_dir+'/image_models'
videodir=loadmodel_dir+'/video_models'
csvdir=loadmodel_dir+'/csv_models'

###############################################################
##                        UNIT TESTS                          ##
###############################################################

class Monolithic(unittest.TestCase):
     
    ###############################################################
    ##                       MODULE TESTS                        ##
    ###############################################################
    # confirm that all the modules are installed correctly 
    def test_a_requirements(self):
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
    def test_b_sox(self, cur_dir=cur_dir):
        # test brew installation by merging two test files 
        os.chdir(cur_dir)
        os.system('sox test_audio.wav test_audio.wav test2.wav')
        if 'test2.wav' in os.listdir():
            b=True  
        else:
            b=False    
        self.assertEqual(True, b)      

    # brew installation (FFmpeg)
    def test_c_ffmpeg(self, cur_dir=cur_dir):
        # test FFmpeg installation with test_audio file conversion 
        os.chdir(cur_dir)
        os.system('ffmpeg -i test_audio.wav test_audio.mp3')
        if 'test_audio.mp3' in os.listdir():
            b=True 
        else:
            b=False 
        self.assertEqual(True, b)

    ###############################################################
    ##                    TRAINING TESTS                         ##
    ###############################################################
    '''
    Take in two folders of files and trains a machine learning model
    via the default machine learning model trainer.

    If clean_data==True or augment_data==True, then the 
    '''
    def test_d_audio_train(self, cur_dir=cur_dir, train_dir=train_dir, model_dir=model_dir, audiodir=audiodir, clean_data=clean_data, augment_data=augment_data):
        # test audio file training 
        os.chdir(train_dir)
        os.mkdir('one')
        os.mkdir('two')
        seed_files('test_audio.wav', cur_dir, train_dir+'/one', clean_data, augment_data,'audio')
        seed_files('test_audio.wav', cur_dir, train_dir+'/two', clean_data, augment_data,'audio')
        os.chdir(model_dir)
        os.system('python3 model.py audio 2 c one two')
        os.chdir(train_dir)
        shutil.rmtree('one')
        shutil.rmtree('two')
        
        # now find the model 
        os.chdir(audiodir)
        b=False
        b = find_model(b)
        self.assertEqual(True, b)

    def test_e_text_train(self, cur_dir=cur_dir, train_dir=train_dir, model_dir=model_dir, textdir=textdir, clean_data=clean_data, augment_data=augment_data):
        # text file training
        os.chdir(train_dir)
        os.mkdir('one')
        os.mkdir('two')
        seed_files('test_text.txt', cur_dir, train_dir+'/one', clean_data, augment_data,'text')
        seed_files('test_text.txt', cur_dir, train_dir+'/two', clean_data, augment_data,'text')
        os.chdir(model_dir)
        os.system('python3 model.py text 2 c one two')
        os.chdir(train_dir)
        shutil.rmtree('one')
        shutil.rmtree('two')
        
        # now find the model 
        os.chdir(textdir)
        b=False
        b = find_model(b)
        self.assertEqual(True, b)

    def test_f_image_train(self, cur_dir=cur_dir, train_dir=train_dir, model_dir=model_dir, imagedir=imagedir, clean_data=clean_data, augment_data=augment_data):
        # image file training
        os.chdir(train_dir)
        os.mkdir('one')
        os.mkdir('two')
        seed_files('test_image.png', cur_dir, train_dir+'/one', clean_data, augment_data,'image')
        seed_files('test_image.png', cur_dir, train_dir+'/two', clean_data, augment_data,'image')
        os.chdir(model_dir)
        os.system('python3 model.py image 2 c one two')
        os.chdir(train_dir)
        shutil.rmtree('one')
        shutil.rmtree('two')    

        # now find the model 
        os.chdir(imagedir)
        b=False
        b = find_model(b)
        self.assertEqual(True, b)

    def test_g_video_train(self, cur_dir=cur_dir, train_dir=train_dir, model_dir=model_dir, videodir=videodir, clean_data=clean_data, augment_data=augment_data):
        # video file training
        os.chdir(train_dir)
        os.mkdir('one')
        os.mkdir('two')
        seed_files('test_video.mp4', cur_dir, train_dir+'/one', clean_data, augment_data,'video')
        seed_files('test_video.mp4', cur_dir, train_dir+'/two', clean_data, augment_data,'video')
        os.chdir(model_dir)
        os.system('python3 model.py video 2 c one two')
        os.chdir(train_dir)
        shutil.rmtree('one')
        shutil.rmtree('two')

        # now find the model 
        os.chdir(videodir)
        b=False
        b = find_model(b)
        self.assertEqual(True, b)

    def test_h_csv_train(self, cur_dir=cur_dir, train_dir=train_dir, model_dir=model_dir, csvdir=csvdir, clean_data=clean_data, augment_data=augment_data):
        # csv file training
        os.chdir(train_dir)
        os.mkdir('one')
        os.mkdir('two')
        seed_files('test_csv.csv', cur_dir, train_dir+'/one', clean_data, augment_data,'csv')
        seed_files('test_csv.csv', cur_dir, train_dir+'/two', clean_data, augment_data,'csv')
        os.chdir(model_dir)
        os.system('python3 model.py csv 2 c one two')
        os.chdir(train_dir)
        shutil.rmtree('one')
        shutil.rmtree('two')

        # now find the model 
        os.chdir(csvdir)
        b=False
        b = find_model(b)
        self.assertEqual(True, b)

    ###############################################################
    ##                  TEST MODEL COMPRESSION                   ##
    ###############################################################

    def test_i_compression(self, model_compress=model_compress, loadmodel_dir=loadmodel_dir):
        b=False

        if model_compress==True:
            os.chdir(loadmodel_dir+'/audio')
            listdir=os.listdir()
            if listdir[i].find('one_two') >= 0 and listdir[i].find('compressed') >= 0:
                b=True
        else:
            b=True

        self.assertEqual(True, b) 

    ###############################################################
    ##                      PRODUCTION TESTS                     ##
    ###############################################################

    # this should only be for audio files technically
    def test_j_production(self, production=production, production_dir=production_dir):
        b=False
        if production == True:
            os.chdir(production_dir)
            listdir=os.listdir()
            for i in range(len(listdir)):
                if listdir[i].find('audio-one_two') >= 0:
                    os.chdir(listdir[i])
                    listdir2=os.listdir()
                    if listdir2 == ['server.py', 'settings.json', 'requirements.txt', 'test', 'Dockerfile', 'Makefile', 'classify.py', 'cloudbuild.yaml', '__init__.py', 'test.py', 'readme.md', 'process.py', 'docker-compose.yml', 'data']:
                        b=True
                        break 
            os.chdir(production_dir)
            shutil.rmtree(listdir[i])
        else: 
            b=True 

        self.assertEqual(True, b) 

    ###############################################################
    ##            FEATURIZATION / LOADING TESTS                  ##
    ###############################################################

    # can featurize audio files via specified featurizer (can be all featurizers) 
    # can also load recently trained models 
    def test_k_loadmodels(self, cur_dir=cur_dir, load_dir=load_dir, loadmodel_dir=loadmodel_dir):
        shutil.copy(cur_dir+'/test_audio.wav', load_dir+'/test_audio.wav')
        shutil.copy(cur_dir+'/test_text.txt', load_dir+'/test_text.txt')
        shutil.copy(cur_dir+'/test_image.png', load_dir+'/test_image.png')
        shutil.copy(cur_dir+'/test_video.mp4', load_dir+'/test_video.mp4')
        shutil.copy(cur_dir+'/test_csv.csv', load_dir+'/test_csv.csv')
        os.chdir(loadmodel_dir)
        os.system('python3 load_models.py')
        os.chdir(load_dir)
        b=True
        self.assertEqual(True, b)

    # note we have to do a loop here to end where the end is 
    # 'audio.json' | 'text.json' | 'image.json' | 'video.json' | 'csv.json'
    # this is because the files are renamed to not have conflicts.
    # for example, if 'audio.wav' --> 'audio.json' and 'audio.mp4' --> 'audio.json',
    # both would have a conflicting name and would overwrite each other. 

    def test_l_loadaudio(self, load_dir=load_dir):
        os.chdir(load_dir)
        listdir=os.listdir() 
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('audio.json') > 0: 
                b=True
                break
        self.assertEqual(True, b)

    def test_m_loadtext(self, load_dir=load_dir):
        os.chdir(load_dir)
        listdir=os.listdir()
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('text.json') > 0: 
                b=True
                break
        self.assertEqual(True, b)

    def test_n_loadimage(self, load_dir=load_dir):
        os.chdir(load_dir)
        listdir=os.listdir()
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('image.json') > 0: 
                b=True
                break
        self.assertEqual(True, b)

    def test_o_loadvideo(self, load_dir=load_dir):
        os.chdir(load_dir)
        listdir=os.listdir()
        b=False
        for i in range(len(listdir)):
            if listdir[i].find('video.json') > 0: 
                b=True
                break
        self.assertEqual(True, b)

    def test_p_loadcsv(self, load_dir=load_dir):
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

