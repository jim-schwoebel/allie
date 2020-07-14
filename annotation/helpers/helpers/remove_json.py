'''
Remove_json.py

Remove all json files in sub-directories.

Useful when you are cloning directories that have already been featurized
to get new feature embeddings with nlx-model repo.
'''
import os 

def removejson(listdir):
    for i in range(len(listdir)):
        if listdir[i][-5:]=='.json':
            os.remove(listdir[i])

listdir=os.listdir()
hostdir=os.getcwd()

for i in range(len(listdir)):
    try:
        os.chdir(hostdir+'/'+listdir[i])
        listdir2=os.listdir()
        removejson(listdir2)

    except:
        pass

