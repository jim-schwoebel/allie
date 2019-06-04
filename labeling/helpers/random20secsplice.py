import soundfile as sf 
import os
import ffmpy
import random
import getpass

genre=input('what folder do you want to create 20 sec splices for?')

dir1='/Users/'+getpass.getuser()+'/Desktop/genres/'+genre
dir2='/Users/'+getpass.getuser()+'/Desktop/genres/'+genre+'_snipped'

os.chdir(dir1)
os.mkdir(dir2)

listdir=os.listdir()

for i in range(len(listdir)):
    try:
        os.chdir(dir1)
        file=listdir[i]
        data, samplerate = sf.read(file)
        totalframes=len(data)
        totalseconds=int(totalframes/samplerate)
        startsec=random.randint(0,totalseconds-21)
        endsec=startsec+20
        startframe=samplerate*startsec
        endframe=samplerate*endsec
        
        #write file to resave wave file at those frames
        os.chdir(dir2)
        sf.write('snipped_'+file, data[int(startframe):int(endframe)], samplerate)
    except:
        print('error, skipping...')
