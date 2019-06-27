
'''
seed_files.py audio 

^^ seed files from command line 
'''
import sys, uuid, os, shutil, time, random
import sounddevice as sd 
import pandas as pd 
import soundfile as sf 
import pyautogui, markovify 

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
        sentence=text_model.make_sentence()
        textfile.write(sentence)

    textfile.close()

def image_record(filename):
    pyautogui.screenshot(filename)

def video_record(filename, test_dir, train_dir):
    print('---------------')
    print('recording video...')
    cur_dur=os.getcwd()
    os.chdir(test_dir+'/helpers/video_record')
    # 3 second recordings 
    os.system('python3 record.py %s 3 %s'%(filename, train_dir))
    os.chdir(cur_dir)
    print('---------------')

def csv_record(filename, newfilename):
    # take in test .CSV and manipulate the columns by copy/paste and re-write 
    csvfile=pd.read_csv(filename)
    filelength=len(filename)
    newlength=random.randint(0,filelength-1)

    # now re-write CSV with the new length 
    g=csvfile.iloc[0:newlength]
    randint2=random.randint(0,1)
    if randint2 == 0:
        g=g+g 
    g.to_csv(newfilename)

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

# get filetype from command line 
filetype=sys.argv[1]
train_dir=sys.argv[2]
cur_dir=os.getcwd()

try:
    os.chdir(train_dir)
except:
    os.mkdir(train_dir)

os.chdir(cur_dir)
# prevdir=prev_dir(cur_dir)
# prevdir=os.getcwd()

if filetype == 'audio':

    '''
    sample command in terminal:
        python3 seed_files.py audio /Users/jimschwoebel/allie/train_dir/one
    '''
    
    # load test data directory 
    if train_dir.endswith('one'):
        data_dir=cur_dir+'/helpers/audio_data/one'
    elif train_dir.endswith('two'):
        data_dir=cur_dir+'/helpers/audio_data/two'

    listdir=os.listdir(data_dir)
    # print(data_dir)
    # option 1 - copy test files
    # --------------------------
    for i in range(len(listdir)):
        if listdir[i][-4:]=='.wav':
            print(listdir[i])
            shutil.copy(data_dir+'/'+listdir[i], train_dir+'/'+listdir[i])
    
    # option 2 - record data yourself (must be non-silent data)
    # --------------------------
    # for i in range(20):
        # filename=str(uuid.uuid4())+'.wav'
        # audio_record(filename, 1, 16000, 1)
elif filetype == 'text':
    '''
    python3 seed_files.py text /Users/jimschwoebel/allie/train_dir/one
    '''
    # Get raw text as string (the Brother's Karamazov)
    with open(cur_dir+'/helpers/text.txt') as f:
        text = f.read()
    # Build the model.
    text_model = markovify.Text(text)
    os.chdir(train_dir)
    for i in range(20):
        filename=str(uuid.uuid4()).replace('-','_')+'.txt'
        text_record(filename, text_model)

elif filetype == 'image':
    '''
    python3 seed_files.py image /Users/jimschwoebel/allie/train_dir/one
    '''
    # take 20 random screenshots with pyscreenshot
    os.chdir(train_dir)
    for i in range(20):
        filename=str(uuid.uuid4()).replace('-','_')+'.png'
        image_record(filename)

elif filetype == 'video':
    '''
    python3 seed_files.py video /Users/jimschwoebel/allie/train_dir/one
    '''
    # make 20 random videos with screenshots 
    os.chdir(train_dir)
    for i in range(20):
        filename=str(uuid.uuid4()).replace('-','_')+'.avi'
        video_record(filename, cur_dir, train_dir)

elif filetype == 'csv':
    '''
    python3 seed_files.py csv /Users/jimschwoebel/allie/train_dir/one
    '''
    # prepopulate 20 random csv files with same headers 
    filename='test_csv.csv'
    shutil.copy(cur_dir+'/'+filename, train_dir+'/'+filename)
    os.chdir(train_dir)
    for i in range(20):
        newfilename=str(uuid.uuid4())+'.csv'
        csv_record(filename, newfilename)
    os.remove(filename)
