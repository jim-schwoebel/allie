import sys, os 
'''
go to the proper folder from cmd line
'''

try:
    foldername=sys.argv[1]
    os.chdir(foldername)
except:
    foldername=input('what folder would you like to delete .JSON files?')
    directory=os.getcwd()
    os.chdir(directory+'/%s'%(foldername))

# remove duplicates using filecmp
listdir=os.listdir()
deleted_files=list()

print('-----------------------------')
print('     DELETING JSON FILES     ')
print('-----------------------------')

for i in range(len(listdir)):
    if listdir[i][-5:]=='.json':
        os.remove(listdir[i])
        deleted_files.append(listdir[i])
 
print('deleted the .JSON files below')
print(deleted_files)
print('-----------------------------')
print('-----------------------------')