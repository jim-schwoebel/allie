import pafy
import os
import pandas as pd
import shutil
import time
import ffmpy
import soundfile as sf
import getpass

#function to clean labels 
def convertlabels(sortlist,labels,textlabels):
    
    clabels=list()
    
    for i in range(len(sortlist)):
        #find index in list corresponding
        index=labels.index(sortlist[i])
        clabel=textlabels[index]
        #pull out converted label
        clabels.append(clabel)

    return clabels 

defaultdir='/Users/'+getpass.getuser()+'/Desktop/downloadaudioset/'
os.chdir(defaultdir)

#load labels of the videos

#number, label, words
loadfile=pd.read_excel('labels.xlsx')

number=loadfile.iloc[:,0].tolist()
labels=loadfile.iloc[:,1].tolist()
textlabels=loadfile.iloc[:,2].tolist()
#remove spaces for folders 
for i in range(len(textlabels)):
    textlabels[i]=textlabels[i].replace(' ','')

#now load data for youtube
loadfile2=pd.read_excel('balanced_train_segments.xlsx')

yid=loadfile2.iloc[:,0].tolist()[2:]
ystart=loadfile2.iloc[:,1].tolist()[2:]
yend=loadfile2.iloc[:,2].tolist()[2:]
#ylabels have to be cleaned to make a good list (CSV --> LIST) 
ylabels_dirty=loadfile2.iloc[:,3].tolist()[2:]
ylabels=list()

#make ylabels list compatible as a list (for shutil copy folder structure later)
for i in range(len(ylabels_dirty)):
    newlabels=ylabels_dirty[i]
    commas=ylabels_dirty[i].count(',')
    if commas==0:
        ylabels.append([ylabels_dirty[i]])
    else:
        listitems=list()
        #loop through and make a list of items 
        while newlabels != '':
            index=newlabels.find(',')
            if index != -1:
                listitem=newlabels[0:index]
                newlabels=newlabels[index+1:]
                listitems.append(listitem)
                commas=newlabels.count(',')
            else:
                listitem=newlabels
                listitems.append(listitem)
                commas=newlabels.count(',')
                newlabels=''
            
        #replace with list of items 
        ylabels.append(list(listitems))
            
        
#make folders
try:
    defaultdir2=os.getcwd()+'/audiosetdata/'
    os.chdir(os.getcwd()+'/audiosetdata')
except:
    defaultdir2=os.getcwd()+'/audiosetdata/'
    os.mkdir(os.getcwd()+'/audiosetdata')
    os.chdir(os.getcwd()+'/audiosetdata')

for i in range(len(textlabels)):
    try:
        os.mkdir(textlabels[i])
    except:
        pass 
        
#iterate through entire CSV file, look for '--' if found, find index, delete section, then go to next index
slink='https://www.youtube.com/watch?v='

for i in range(len(yid)):
    link=slink+yid[i]
    start=ystart[i]
    end=yend[i]
    clabels=convertlabels(ylabels[i],labels,textlabels)

    for j in range(len(clabels)):
        
        #change to the right directory
        newdir=defaultdir2+clabels[j]+'/'
        os.chdir(newdir)
        
        if j ==0:
            
            #if it is the first download, pursue this path to download video 
            lastdir=os.getcwd()+'/'
    
            try:
                video=pafy.new(link)
                bestaudio=video.getbestaudio()
                filename=bestaudio.download()
                extension=bestaudio.extension
                #get file extension and convert to .wav for processing later 
                os.rename(filename,'%s_start_%s_end_%s%s'%(str(i),start,end,extension))
                filename='%s_start_%s_end_%s%s'%(str(i),start,end,extension)
                if extension not in ['.wav']:
                    xindex=filename.find(extension)
                    filename=filename[0:xindex]
                    ff=ffmpy.FFmpeg(
                        inputs={filename+extension:None},
                        outputs={filename+'.wav':None}
                        )
                    ff.run()
                    os.remove(filename+extension)
                
                file=filename+'.wav'
                data,samplerate=sf.read(file)
                totalframes=len(data)
                totalseconds=totalframes/samplerate
                startsec=start
                startframe=samplerate*startsec
                endsec=end
                endframe=samplerate*endsec
                sf.write('snipped'+file, data[startframe:endframe], samplerate)
                snippedfile='snipped'+file
                os.remove(file)
                
            except:
                print('no urls')

        else:
            #copy if already downloaded to proper labeled directory
            #this will eliminated repeated youtube calls to download
            print('copying file to %s'%(newdir+snippedfile))
            try:
                shutil.copy(lastdir+snippedfile,newdir+snippedfile)
            except:
                print('error copying file')

    #sleep 5 seconds to prevent IP from getting banned 
    time.sleep(5) 


