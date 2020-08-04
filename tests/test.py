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


|_   _|       | |    / _ \ | ___ \_   _|
  | | ___  ___| |_  / /_\ \| |_/ / | |  
  | |/ _ \/ __| __| |  _  ||  __/  | |  
  | |  __/\__ \ |_  | | | || |    _| |_ 
  \_/\___||___/\__| \_| |_/\_|    \___/ 
  
Use this script for unit testing of Allie's functionality. 

In particular:
- class test_dependencies(unittest.TestCase) 
tests for modules and brew installations (FFmpeg and SoX).
- class test_cleaning(unittest.TestCase)
tests ability to clean files via cleaning scripts (mostly de-duplication, will expand in future).
- class test_augmentation(unittest.TestCase)
tests ability to augment files via augmentation scripts (in ./datasets/) directory.
- class test_features(unittest.TestCase)
tests ability to featurize files via default_featurizers.
- class test_transcription(unittest.TestCase) 
tests ability to transcribe files.
- class test_training(unittest.TestCase)
tests ability to train machine learning models (classification and regression) with all settings.
- class test_preprocessing(unittest.TestCase)
tests ability to create transformations with the transform.py script (for model training).
- class test_loading(unittest.TestCase)
tests ability to load model files and make predictions via model directory (via test files / load_dir / models trained).
- class test_visualization(unittest.TestCase)
tests ability to visualize classification problems through the visualize.py script.

Note that these unit tests are contextual, meaning that only the settings specified 
in settings.json are tested. This makes it easy to quickly test whether or not your 
current computer and operating system can handle new settings that you specify within 
Allie's core API.
'''
import os, time, shutil

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

def remove_temp_model():
    '''removes temporary model files'''
    listdir=os.listdir()
    for i in range(len(listdir)):
        if listdir[i].find('one_two') == 0:
            os.remove(listdir[i])

###############################################################
##                    GET FOLDER INFO.                       ##
###############################################################
cur_dir = os.getcwd()
prevdir= prev_dir(cur_dir)
load_dir = prevdir+'/load_dir'
train_dir = prevdir + '/train_dir'
model_dir = prevdir+ '/training'
loadmodel_dir = prevdir+'/models'

# remove one and two directories if they exist in train_dir to allow for 
# proper testing.
os.chdir(train_dir)
listdir=os.listdir()
if 'one' in listdir:
    shutil.rmtree('one')
if 'two' in listdir:
    shutil.rmtree('two')

os.chdir(cur_dir)

###############################################################
##                    RUN UNIT TESTS.                        ##
###############################################################
os.system('python3 unit_test.py')

###############################################################
##                    REMOVE TEMP FILES                      ##
###############################################################
print('-----------------^^^-----------------------')
print('-------------^^^^---^^^^-------------------')
print('-----------CLEANUP TEMP FILES--------------')
print('---------^^^^^^^^^^^^^^^^^^^^^^------------')
os.chdir(cur_dir)
print('deleting temp files from FFmpeg and SoX tests')
print('-------------------------------------------')
try:
	os.remove('test2.wav')
except:
	print('test2.wav does not exist, cannot delete it.')
try:
	os.remove('test_audio.mp3')
except:
	print('test_audio.wav does not exist, cannot delete it.')

# now we can remove everything in load_dir
os.chdir(load_dir)
listdir=os.listdir()

# remove everything in the load_dir (to allow for future non-conflicting featurizations / model predictions)
print('deleting temp files load_dir tests')
print('-------------------------------------------')
for i in range(len(listdir)):
    if listdir[i].endswith('.json') or listdir[i].endswith('.wav') or listdir[i].endswith('.png') or listdir[i].endswith('.txt') or listdir[i].endswith('.csv') or listdir[i].endswith('.mp4'):
        os.remove(listdir[i])

print('deleting temp model files (audio, text, image, and video)')
print('-------------------------------------------')
# we can also remove all temporarily trained machine learning models 
try:
    os.chdir(loadmodel_dir+'/audio_models')
    remove_temp_model()
except:
    pass 

try:
    os.chdir(loadmodel_dir+'/text_models')
    remove_temp_model()
except:
    pass 

try:
    os.chdir(loadmodel_dir+'/image_models')
    remove_temp_model()
except:
    pass 

try:
    os.chdir(loadmodel_dir+'/video_models')
    remove_temp_model()
except:
    pass 

try:
    os.chdir(loadmodel_dir+'/csv_models')
    remove_temp_model()
except:
    pass 

os.chdir(train_dir)
try:
    shutil.rmtree('audio_features')
except:
    pass
try:
    shutil.rmtree('text_features')
except:
    pass 
try:
    shutil.rmtree('image_features')
except:
    pass 
try:
    shutil.rmtree('video_features')
except:
    pass
try:
    shutil.rmtree('csv_features')
except:
    pass
try:
    shutil.rmtree('audio_transcription')
except:
    pass
try:
    shutil.rmtree('text_transcription')
except:
    pass
try:
    shutil.rmtree('image_transcription')
except:
    pass
try:
    shutil.rmtree('video_transcription')
except:
    pass
try:
    shutil.rmtree('csv_transcription')
except:
    pass
try:
    shutil.rmtree('audio_augmentation')
except:
    pass
try:
    shutil.rmtree('image_augmentation')
except:
    pass
try:
    shutil.rmtree('text_augmentation')
except:
    pass
try:
    shutil.rmtree('video_augmentation')
except:
    pass
try:
    shutil.rmtree('csv_augmentation')
except:
    pass
