import filecmp, os,time, platform, json, datetime 

foldername='voice_samples'
directory=os.getcwd()
os.chdir(directory+'/%s'%(foldername))

# remove duplicates using filecmp
listdir=os.listdir()
unclean_filelist=listdir 
deleted_files=list()

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
            print('error, moving on...')
            
#calculate metadata
processtime=time.time()-start 
clean_filelist=os.listdir()

#write .json output of the session in the crypto-cli folder as a new folder: 'decrypted_processed_metadata'
os.chdir(directory)

data={
    'date':str(datetime.datetime.now()),
    'downloaded AWS folder':bucket,
    'processtime':str(processtime),
    'clean filelist':clean_filelist,
    'unclean filelist':unclean_filelist,
    'deleted files':deleted_files,
    'operating system':platform.system(),
    'os release':platform.release(),
    'os version':platform.version(),
    }

jsonfilename=foldername+'.json'
jsonfile=open(jsonfilename,'w')
json.dump(data,jsonfile)
jsonfile.close()