'''
clean.py [folderpath]

This will take in a folder, classify that folder as 
audio, text, image, video, or .CSV files, and then
clean the dataset appropriately
'''
import os, sys

def classifyfolder(listdir):
    filetypes=list()
    for i in range(len(listdir)):
        if listdir[i].endswith(('.mp3', '.wav')):
            filetypes.append('audio')
        elif listdir[i].endswith(('.png', '.jpg')):
            filetypes.append('image')
        elif listdir[i].endswith(('.txt')):
            filetypes.append('text')
        elif listdir[i].endswith(('.mp4', '.avi')):
            filetypes.append('video')
        elif listdir[i].endswith(('.csv')):
            filetypes.append('csv')

    counts={'audio': filetypes.count('audio'),
            'image': filetypes.count('image'),
            'text': filetypes.count('text'),
            'video': filetypes.count('video'),
            'csv': filetypes.count('csv')}

    # get back the type of folder (main file type)
    filetypes=list(counts)
    values=list(counts.values())
    index=values.index(max(values))

    return filetypes[index]

try:
	folderpath=sys.argv[2]
except:
	folderpath=input('what is the folder path you would like to clean? \n (e.g. /Users/jimschwoebel/allie/train_dir/one) \n')

try: 
    modelpath=sys.argv[1]
except:
    modelpath=input('what is the model directory? \n (e.g. /Users/jimschwoebel/allie/datasets/cleaning/models) \n')

curdir=os.getcwd()
os.chdir(folderpath)
listdir=os.listdir()
problemtype=classifyfolder(listdir)
print(problemtype + 'folder detected!')
print('-----------------')
os.chdir(curdir)

################################################
##         UNIVERSAL CLEANING SCRIPTS         ## 
################################################

# delete file duplicates (applies to all file tpyes)
os.system('python3 delete_duplicates.py %s'%(folderpath))
os.system('python3 delete_json.py %s'%(folderpath))

################################################
##       FILE-SPECIFIC CLEANING SCRIPTS       ## 
################################################

if problemtype=='audio':
    # the order here is important 
    # delete files that have multiple speakers, remove silence, then normalize volumes.
    os.system('python3 audio/delete_multi_speaker.py %s %s'%(modelpath, folderpath))
    os.system('python3 audio/remove_silence.py %s'%(folderpath))
    os.system('python3 audio/normalize_volume.py %s'%(folderpath))
elif problemtype=='image':
    pass
elif problemtype=='text':
    pass
elif problemtype=='video':
    pass
elif problemtype=='csv':
    pass 
