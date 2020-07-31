import os, random
import soundfile as sf 

def augment_randomsplice(filename):

    # random10secondsplice
    slicenum=1

    file=filename
    data, samplerate = sf.read(file)
    totalframes=len(data)
    totalseconds=int(totalframes/samplerate)
    startsec=random.randint(0,totalseconds-(slicenum+1))
    endsec=startsec+slicenum
    startframe=samplerate*startsec
    endframe=samplerate*endsec
    sf.write('snipped%s_'%(str(slicenum))+file, data[int(startframe):int(endframe)], samplerate)
