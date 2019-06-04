'''
Make_controls.py

Generate control data from a list of folders filled with .wav files.
'''
import soundfile as sf 
import os, ffmpy, random, shutil

# CONVERT FILE

def convert_file(filename):
    
    #take in an audio file and convert with ffpeg file type
    #types of input files: .ogg 
    #output file type: .wav
    newfile=filename[0:-4]+'.wav'
    
    ff = ffmpy.FFmpeg(
        inputs={filename:None},
        outputs={newfile: None}
        )
    ff.run()

    os.remove(filename)
    
    return newfile

##############################################################################

default_dir=os.getcwd()+'/'
os.chdir(default_dir)
g=os.listdir()
h=0

##############################################################################
# CONVERT ALL FILES TO WAV (COMMENTED OUT)

convertfiles='n'

if convertfiles in ['y','yes']:

   #check first if all the files are .wav files and if not convert them and delete the other file type
   for i in range(len(g)):
       try:
           if g[i] not in ['.DS_Store']:
               os.chdir(default_dir+g[i])
               h=os.listdir()
               for j in range(len(h)):
                   try:
                       if h[j][-4:]!='.wav':
                           print('converting %s'%(h[j]))
                           new_file=convert_file(h[j])
                   except:
                       print('error')
       except:
           print('error')

else:
   pass

##############################################################################

class_default=input('what is the class that will not be used as a control?')
control_dir=class_default+'_controls'
os.mkdir(control_dir)

os.chdir(default_dir+class_default)
q=os.listdir()
filenum=len(q)
movedlist=list()
movedlist.append('')
ind=g.index('.DS_Store')
if ind>=0:
    del g[ind]
else:
    pass

#need equal amount of controls
#loop over total number of files in class over the number of classes -1 (not including class)

count=0
for i in range(int(len(q)/(len(g)-1))):

    for j in range(len(g)):

        print(g[j])
        if g[j] not in [class_default]:
            
            try:
            
                print('changing to %s directory'%(default_dir+g[j]))
                os.chdir(default_dir+g[j])
                h=os.listdir()
                file_num=len(h)
                cur_file=''
                count=0
                while cur_file in movedlist:
                    if count>file_num:
                        break
                    
                    randint=random.randint(0,file_num-1)
                    cur_file=h[randint]
                    count=count+1

                print('copying file: %s'%(cur_file))
                shutil.copy(default_dir+g[j]+'/'+cur_file,default_dir+control_dir+'/'+cur_file)
                movedlist.append(cur_file)
                count=count+1 

            except:
                print('error')

if count==0:

    for j in range(len(g)):

        print(g[j])
        if g[j] not in [class_default]:
            
            try:
            
                print('changing to %s directory'%(default_dir+g[j]))
                os.chdir(default_dir+g[j])
                h=os.listdir()
                file_num=len(h)
                cur_file=''
                count=0
                while cur_file in movedlist:
                    if count>file_num:
                        break
                    
                    randint=random.randint(0,file_num-1)
                    cur_file=h[randint]
                    count=count+1

                print('copying file: %s'%(cur_file))
                shutil.copy(default_dir+g[j]+'/'+cur_file,default_dir+control_dir+'/'+cur_file)
                movedlist.append(cur_file)
                count=count+1 

            except:
                print('error')

