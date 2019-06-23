'''
augment.py [folderpath]

This will take in a folder, classify that folder as 
audio, text, image, video, or .CSV files, and then
augment the dataset appropriately.
'''
import os, sys, random, shutil, time 

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
	folderpath=sys.argv[1]
except:
	folderpath=input('what is the folder path you would like to augment? \n (e.g. /Users/jimschwoebel/allie/train_dir/one) \n')

curdir=os.getcwd()
os.chdir(folderpath)
folderpath=os.getcwd()
listdir=os.listdir()
problemtype=classifyfolder(listdir)
os.chdir(curdir)

################################################
##         UNIVERSAL AUGMENTATION SCRIPTS      ## 
################################################

# delete file duplicates (applies to all file tpyes)
random.shuffle(listdir)
augment_files=list()
cur_dir=os.getcwd()
temp_dir=folderpath+'/temp_folder'
try:
    os.mkdir(temp_dir)
except:
    shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

for i in range(int(len(listdir)/3)):
    # find 33% of the dataset to augment 
    augment_files.append(listdir[i])
    shutil.move(folderpath+'/'+listdir[i], temp_dir+'/'+listdir[i])

################################################
##     FILE-SPECIFIC AUGMENTATION SCRIPTS     ## 
################################################

print(temp_dir)
if problemtype=='audio':
    # audio augmentation
    os.chdir(curdir+'/audio_augmentation')
    print(os.getcwd())
    os.system('python3 audio_augment.py %s'%(temp_dir))
elif problemtype=='image':
    print('image augmentation coming soon! passing now...')
elif problemtype=='text':
    print('text augmentation coming soon! passing now...')
elif problemtype=='video':
    print('video augmentation coming soon! passing now...')
elif problemtype=='csv':
    print('csv augmentation coming soon! passing now...') 

# now move all the files in the temp directory back into main directory
os.chdir(temp_dir)
listdir=os.listdir()
for i in range(len(listdir)):
    shutil.move(temp_dir+'/'+listdir[i], folderpath+'/'+listdir[i])
os.chdir(folderpath)
shutil.rmtree('temp_folder')
