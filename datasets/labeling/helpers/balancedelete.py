import os, random, shutil

## helper functions 

def get_wav():
    # get all .WAV or .MP3 files in folder and count the number of them 
    listdir=os.listdir()
    count=0
    for i in range(len(listdir)):
        if listdir[i][-4:] in ['.wav', '.mp3']:
            count = count+1 
    return count 

def random_remove(remove_num):
    # remove a number of files to balnace classes.
    listdir=os.listdir()
    wavfiles=list()
    random.shuffle(listdir)
    for i in range(len(listdir)):
        if listdir[i][-4:] in ['.wav', '.mp3']:
            wavfiles.append(listdir[i])
    for i in range(remove_num):
        os.remove(wavfiles[i])
    print('removed %s .wav or .mp3 files'%(remove_num))


# now go to main script 
listdir=os.listdir()
# find all folders
folders=list()
for i in range(len(listdir)):
    if listdir[i].find('.') < 0:
        folders.append(listdir[i])

curdir=os.getcwd()
counts=list()
for i in range(len(folders)):
    os.chdir(curdir)
    os.chdir(folders[i])
    count=get_wav()
    counts.append(count)

# now find minimum
min_=min(counts)

for i in range(len(folders)):
    os.chdir(curdir)
    os.chdir(folders[i])
    count=get_wav()
    if count > min_:
        remove_num=count-min_
        random_remove(remove_num)


                    
