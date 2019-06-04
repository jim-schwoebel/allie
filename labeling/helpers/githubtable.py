'''
Load all model accuracies, names, and standard deviations
and output them in a spreadsheet.

This is intended for any model file directory using the nlx-model repository.'''

import json, os, xlsxwriter, getpass

def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
     
    return z

os.chdir(os.getcwd()[0:-(len('nlx-datalabeling'))]+'nlx-audiomodel/models')

listdir=os.listdir()

names=list()
accs=list()
stds=list()
modeltypes=list()

for i in range(len(listdir)):
    if listdir[i][-5:]=='.json':
        try:
            g=json.load(open(listdir[i]))
            acc=g['accuracy']
            name=g['model']
            std=g['deviation']
            modeltype=g['modeltype']

            names.append(name)
            accs.append(acc)
            stds.append(std)
            modeltypes.append(modeltype)
        except:
            print('error %s'%(listdir[i]))

names=sort_list(names, accs)
stds=sort_list(stds, accs)
modeltypes=sort_list(modeltypes, accs)
accs=sort_list(accs, accs)

file = open('table.txt','w')
file.write('| '+'Model Name' + ' |')
file.write(' Accuracy'+' |')
file.write(' Standard Deviation' + ' |')
file.write(' Modeltype'+ ' |')
file.write('\n')

file.write('|-----|-----|-----|-----|')
file.write('\n')

print(names)
for j in range(len(names)):
    file.write('| '+str(names[j])+' |')
    file.write(' '+str(accs[j])+' |')
    file.write(' '+str(stds[j])+' |')
    file.write(' '+str(modeltypes[j])+' |')
    file.write('\n')

file.close()

os.system('open %s'%(os.getcwd()+'/table.txt'))
