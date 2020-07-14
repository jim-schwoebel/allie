import sys, os, shutil, librosa, uuid
from pyvad import vad, trim, split
import matplotlib.pyplot as plt
import numpy as np

# make sure the right version of numba is installed
os.system('pip3 install numba==0.48')

def clean_file(wavfile):
    '''
    taken from https://github.com/F-Tag/python-vad/blob/master/example.ipynb
    '''
    show=False
    curdir=os.getcwd()
    data, fs = librosa.core.load(wavfile)
    time = np.linspace(0, len(data)/fs, len(data))
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

    # pretrimmed
    fig, ax1 = plt.subplots()
    ax1.plot(time, data, label='speech waveform')
    ax1.set_xlabel("TIME [s]")
    ax2=ax1.twinx()
    ax2.plot(time, vact, color="r", label = 'vad')
    plt.yticks([0, 1] ,('unvoice', 'voice'))
    ax2.set_ylim([-0.01, 1.01])
    plt.legend()
    plt.savefig('voiced_unvoiced.png')
    if show==True:
        plt.show()
    plt.close()

    folder=str(uuid.uuid4())
    os.mkdir(folder)
    os.chdir(folder)
    command='sox'
    for i in range(len(utterances)):
        
        trimmed = data[utterances[i][0]:utterances[i][1]]
        time = np.linspace(0, len(trimmed)/fs, len(trimmed)) # time axis
        fig, ax1 = plt.subplots()
        ax1.plot(time, trimmed, label='speech waveform')
        ax1.set_xlabel("TIME [s]")
        if show==True:
            plt.show()
        plt.close()

        # now overwrite file
        librosa.output.write_wav('%s.wav'%(str(i)), trimmed, fs)
        command=command+' %s'%(str(i)+'.wav')
    command=command+' '+wavfile
    os.system(command)
    os.chdir(curdir)
    os.remove(wavfile)
    shutil.copy(os.getcwd()+'/%s/'%(folder)+wavfile, os.getcwd()+'/'+wavfile)
    shutil.rmtree(folder)
    
wavfile=sys.argv[1]
clean_file(wavfile)