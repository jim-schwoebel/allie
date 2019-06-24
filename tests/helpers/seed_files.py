
'''
seed_files.py audio 

^^ seed files from command line 
'''
import sys, uuid, os 
import sounddevice as sd 
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
        textfile.write(text_model.make_sentence())

    textfile.close()

def image_record(filename):
    pyautogui.screenshot(filename)

def video_record(filename, test_dir):
    print('---------------')
    print('recording video...')
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

# get filetype from command line 
filetype=sys.argv[1]
cur_dir=os.getcwd()

if filetype == 'audio':
    # load test data directory 
    if train_dir.endswith('one'):
        data_dir+cur_dir+'/helpers/audio_data/one'
    elif train_dir.endswith('two'):
        data_dir+cur_dir+'/helpers/audio_data/two'

    listdir=os.listdir(data_dir)
    # print(data_dir)
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
        text_record(filename, text_model)

elif filetype == 'image':
    # take 20 random screenshots with pyscreenshot
    for i in range(20):
        filename=str(uuid.uuid4())+'.png'
        image_record(filename)

elif filetype == 'video':
    # make 20 random videos with screenshots 
    for i in range(20):
        filename=str(uuid.uuid4()).replace('-','_')+'.avi'
        video_record(filename, cur_dir)

elif filetype == 'csv':
    # prepopulate 20 random csv files with same headers 
    shutil.copy(cur_dir+'/'+filename, train_dir+'/'+filename)
    for i in range(20):
        newfilename=str(uuid.uuid4())+'.csv'
        csv_record(filename, newfilename)
    os.remove(filename)
