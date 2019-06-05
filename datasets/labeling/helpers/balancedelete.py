import os
import random
import shutil

folder1=input('path to folder1')
folder2=input('path to folder2')

#wavfiles

os.chdir(folder1)
listdir1=os.listdir()
filelist1=list()
for i in range(len(listdir1)):
    if listdir1[i][-4:]=='.wav':
        filelist1.append(listdir1[i])

os.chdir(folder2)
listdir2=os.listdir()
filelist2=list()
for i in range(len(listdir2)):
    if listdir2[i][-4:]=='.wav':
        filelist2.append(listdir2[i])

l1=len(filelist1)
l2=len(filelist2)

if l1 > l2:
    dif=l1-l2
    print('Folder 1 has %s more wave files than folder 2. Deleting files in folder 1 to balance.'%(str(dif)))
    os.chdir(folder1)
    removelist=list()
    for i in range(dif):
        randnum=random.randint(0,len(filelist1)-1)
        print('removing %s'%(filelist1[randnum]))
        os.remove(filelist1[randnum])
        filelist1.remove(filelist1[randnum])
    
elif l2 > l1:
    dif=l2-l1
    print('Folder 2 has %s more wave files than folder 1. Deleting files in folder 2 to balance.'%(str(dif)))
    os.chdir(folder2)
    removelist=list()
    for i in range(dif):
        randnum=random.randint(0,len(filelist2)-1)
        print('removing %s'%(filelist2[randnum]))
        os.remove(filelist2[randnum])
        filelist2.remove(filelist2[randnum])
else:
    print('Both folders have the same number of wave files, already. Passing...')

                    
