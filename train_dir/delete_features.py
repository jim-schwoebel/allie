'''
delete_features.py

Clear all json files from folders
'''
import os, json
from tqdm import tqdm

sampletype=sys.argv[1]
feature_set=sys.argv[2]

folders=list()
listdir=os.listdir()
for i in range(len(listdir)):
	if listdir[i].find('.')<0:
		folders.append(listdir[i])

print(folders)

# remove all json files
curdir=os.getcwd()
for i in tqdm(range(len(folders))):
        os.chdir(curdir)
        os.chdir(folders[i])
        listdir=os.listdir()
        for i in range(len(listdir)):

                if listdir[i].endswith('.json'):
                        # print(listdir[i])
                        # os.remove(listdir[i])
                        data=json.load(open(listdir[i]))
                        del data['features'][sampletype][feature_set]
                        jsonfile=open(listdir[i],'w')
                        json.dump(data,jsonfile)
                        jsonfile.close()
