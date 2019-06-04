import librosa
import os
import soundfile as sf
import xlsxwriter
import pandas as pd
import time
import json 

exceldirectory='/Users/jim/Desktop/neurolex/voicemails/'
jsondirectory='/Users/jim/Desktop/neurolex/voicemails/jsonfiles/'
jsonexceldirectory='/Users/jim/Desktop/neurolex/voicemails/jsonfiles-excel/'

os.chdir(exceldirectory)

ls=pd.read_excel('voicemails.xlsx')

filename=ls.iloc[:,0]
#name of file to align with json files 
gender=ls.iloc[:,2]
#0=music, 1=male, 2=femaile, 3=male/female, 4=child, 5=multi-child)
age=ls.iloc[:,3]
#0=music, 1=adult, 2=child
sadness=ls.iloc[:,4]
#0=music, 1=least, 10=most
happiness=ls.iloc[:,5]
#0=music, 1=least, 10=most
stress=ls.iloc[6]
#0=music, 1=lest, 10=most
dialect=ls.iloc[7]
#0=music, 1=american dialect, 2=foreign dialect
voicemusic=ls.iloc[8]
#1=voice, 2=music, 3=multi-music and voice
fatigue=ls.iloc[9]
#0=music, 1=least, 10=most
audioquality=ls.iloc[10]
#0=nothing, 1=lowest, 10=highest
sickness=ls.iloc[11]
#1=natural, 2=non-natural, 3=music, 4=sick

##os.chdir(jsondirectory)
##jsonfiles=os.listdir()

os.chdir(jsonexceldirectory)
 
for g in range(len(filename)):
    
    #try:
    #find json file recording ID field 
##    jsonfileread=open(jsonfiles[i],'r').read()
##    jsonfile=json.loads(jsonfileread)
##    jsonfileid=jsonfile['recordingID']
##    #search for this recording ID in excel file 
##    for g in range(len(filename)):
##        if filename[g]==jsonfileid:
##            indval=int(g)

    #looks for the index of the filename matching row
    #creates array of new data to add to json 
    newdata={
        'filename':filename[g],
        'gender': int(gender[g]),
        'age': int(age[g]),
        'sadness': int(sadness[g]),
        'happiness': int(happiness[g]),
        'stress': stress[g],
        'dialect': int(dialect[g]),
        'voicemusic': int(voicemusic[g]),
        'fatigue': int(fatigue[g]),
        'audioquality': int(audioquality[g]),
        'sickness': int(sickness[g]),
        }

    #dump to new directory 
   
    json.dump(jsonfile)

    #except:
        #if not in excel file and no match, print this 
       # print('no file match found')
    
