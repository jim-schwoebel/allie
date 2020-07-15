  
import soundfile as sf 
import os, ffmpy, random, getpass

def clean_random20secsplice(audiofile):
    try:
        data, samplerate = sf.read(audiofile)
        totalframes=len(data)
        totalseconds=int(totalframes/samplerate)
        startsec=random.randint(0,totalseconds-21)
        endsec=startsec+20
        startframe=samplerate*startsec
        endframe=samplerate*endsec
        
        #write file to resave wave file at those frames
        sf.write('snipped_'+audiofile, data[int(startframe):int(endframe)], samplerate)
        os.remove(audiofile)
    except:
        print('error, skipping...')