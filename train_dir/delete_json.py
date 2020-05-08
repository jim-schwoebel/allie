'''
delete_json.py

Clear all json files from folders
'''
import os

folders=list()
listdir=os.listdir()
for i in range(len(listdir)):
	if listdir[i].find('.')<0:
		folders.append(listdir[i])

print(folders)

# remove all json files
curdir=os.getcwd()
for i in range(len(folders)):
        os.chdir(curdir)
        os.chdir(folders[i])
        listdir=os.listdir()
        for i in range(len(listdir)):
                if listdir[i].endswith('.json'):
                        print(listdir[i])
                        os.remove(listdir[i])
