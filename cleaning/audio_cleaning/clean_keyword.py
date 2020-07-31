import sys, os, shutil, librosa, uuid
from pyvad import vad, trim, split
import matplotlib.pyplot as plt
import numpy as np

# make sure the right version of numba is installed
os.system('pip3 install numba==0.48')

def transcribe_audiofile(file):

    curdir=os.getcwd()
    listdir=os.listdir()
    deepspeech_dir=os.getcwd()

    # download models if not in helper directory
    if 'deepspeech-0.7.0-models.pbmm' not in listdir:
        os.system('wget https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.pbmm')
    if 'deepspeech-0.7.0-models.scorer' not in listdir:
        os.system('wget https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.scorer')

    # initialize filenames
    textfile=file[0:-4]+'.txt'
    newaudio=file[0:-4]+'_newaudio.wav'

    if deepspeech_dir.endswith('/'):
        deepspeech_dir=deepspeech_dir[0:-1]

    # go back to main directory
    os.chdir(curdir)

    # try:
    # convert audio file to 16000 Hz mono audio 
    os.system('ffmpeg -i "%s" -acodec pcm_s16le -ac 1 -ar 16000 "%s" -y'%(file, newaudio))
    command='deepspeech --model %s/deepspeech-0.7.0-models.pbmm --scorer %s/deepspeech-0.7.0-models.scorer --audio "%s" >> "%s"'%(deepspeech_dir, deepspeech_dir, newaudio, textfile)
    print(command)
    os.system(command)

    # get transcript
    transcript=open(textfile).read().replace('\n','')
    # remove temporary files
    os.remove(textfile)
    os.remove(newaudio)
    # except:
    #     try:
    #         # remove temporary files
    #         os.remove(textfile)
    #     except:
    #         pass
    #     try:
    #         os.remove(newaudio)
    #     except:
    #         pass
    #     transcript=''

    return transcript 


def clean_keyword(audiofile,keyword):
    '''
    taken from https://github.com/F-Tag/python-vad/blob/master/example.ipynb
    '''
    show=False
    curdir=os.getcwd()
    data, fs = librosa.core.load(audiofile)
    time = np.linspace(0, len(data)/fs, len(data))
    try:
        vact = vad(data, fs, fs_vad = 16000, hop_length = 30, vad_mode=3)
        vact = list(vact)
        while len(time) > len(vact):
            vact.append(0.0)
        utterances=list()

        for i in range(len(vact)):
            try:
                if vact[i] != vact[i-1]:
                    # voice shift 
                    if vact[i] == 1:
                        start = i
                    else:
                        # this means it is end 
                        end = i
                        utterances.append([start,end])
            except:
                pass
        
        print(utterances)
        vact=np.array(vact)

        tempfiles=list()

        for i in range(len(utterances)):
            trimmed = data[utterances[i][0]:utterances[i][1]]
            tempfile = str(uuid.uuid4())+'.wav'
            librosa.output.write_wav(tempfile, trimmed, fs)
            tempfiles.append(tempfile)

        for i in range(len(tempfiles)):
            if os.path.getsize(tempfiles[i]) > 20000:
                pass
                # transcript=transcribe_audiofile(tempfiles[i])
                # print('TRANSCRIPT --> %s'%(transcript))
                # if transcript == 'coconut':
                   #  pass
               #  else:
                  #  os.remove(tempfiles[i])
            else:
                os.remove(tempfiles[i])
            # if transcript==keyword:
            #     
            #     pass
            # else:
            #     os.remove(tempfiles[i])
    except:
        print('ERROR - ValueError: When data.type is float, data must be -1.0 <= data <= 1.0.')

    os.remove(audiofile)