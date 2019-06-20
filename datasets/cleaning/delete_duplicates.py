import filecmp, os,time, platform, json, datetime 
import sys

try:
    foldername=sys.argv[1]
    os.chdir(foldername)
except:
    foldername=input('what folder would you like to delete duplicates?')
    directory=os.getcwd()
    os.chdir(directory+'/%s'%(foldername))

# remove duplicates using filecmp
listdir=os.listdir()
unclean_filelist=listdir 
deleted_files=list()

print('-----------------------------')
print('     DELETING DUPLICATES     ')
print('-----------------------------')

for i in range(len(listdir)):
    file=listdir[i]
    listdir2=os.listdir()

    #now sub-loop through all files in directory and remove duplicates 
    for j in range(len(listdir2)):
        try:
            if listdir2[j]==file:
                pass
            elif listdir2[j]=='.DS_Store':
                pass 
            else:
                if filecmp.cmp(file, listdir2[j])==True:
                    print('removing duplicate: %s ____ %s'%(file,listdir2[j]))
                    deleted_files.append(listdir2[j])
                    os.remove(listdir2[j])
                else:
                    pass
        except:
            pass 
 
