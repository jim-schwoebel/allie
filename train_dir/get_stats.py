'''
               AAA               lllllll lllllll   iiii                      
              A:::A              l:::::l l:::::l  i::::i                     
             A:::::A             l:::::l l:::::l   iiii                      
            A:::::::A            l:::::l l:::::l                             
           A:::::::::A            l::::l  l::::l iiiiiii     eeeeeeeeeeee    
          A:::::A:::::A           l::::l  l::::l i:::::i   ee::::::::::::ee  
         A:::::A A:::::A          l::::l  l::::l  i::::i  e::::::eeeee:::::ee
        A:::::A   A:::::A         l::::l  l::::l  i::::i e::::::e     e:::::e
       A:::::A     A:::::A        l::::l  l::::l  i::::i e:::::::eeeee::::::e
      A:::::AAAAAAAAA:::::A       l::::l  l::::l  i::::i e:::::::::::::::::e 
     A:::::::::::::::::::::A      l::::l  l::::l  i::::i e::::::eeeeeeeeeee  
    A:::::AAAAAAAAAAAAA:::::A     l::::l  l::::l  i::::i e:::::::e           
   A:::::A             A:::::A   l::::::ll::::::li::::::ie::::::::e          
  A:::::A               A:::::A  l::::::ll::::::li::::::i e::::::::eeeeeeee  
 A:::::A                 A:::::A l::::::ll::::::li::::::i  ee:::::::::::::e  
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee  
                                                                             
|  \/  |         | |    | |  / _ \ | ___ \_   _|
| .  . | ___   __| | ___| | / /_\ \| |_/ / | |  
| |\/| |/ _ \ / _` |/ _ \ | |  _  ||  __/  | |  
| |  | | (_) | (_| |  __/ | | | | || |    _| |_ 
\_|  |_/\___/ \__,_|\___|_| \_| |_/\_|    \___/ 
						   
Makes a table in Microsoft Word for all the audio features present in a file in a particular folder.
This is useful for peer-reviewed publications (for supplementary tables).

Usage: python3 get_stats.py [folder]

Example: python3 get_stats.py females

Following this tutorial with modifications: https://towardsdatascience.com/how-to-generate-ms-word-tables-with-python-6ca584df350e
'''

from docx import Document
from docx.shared import Cm, Pt
import numpy as np
import os, json, time, sys

def describe_text(jsonfile):

	# get dictionary 
	g=json.load(open(jsonfile))
	features=g['features']['audio']
	featuretypes=list(features)

	print(featuretypes)
	features_=list()
	labels_=list()
	for j in range(len(featuretypes)):
		rename_labels=list()
		temp_labels=features[featuretypes[j]]['labels']
		for k in range(len(temp_labels)):
			rename_labels.append(temp_labels[k]+' (%s)'%(featuretypes[j]))
		try:
			features_=features_+features[featuretypes[j]]['features']
		except:
			features_=features_+[0]
		labels_=labels_+rename_labels

	description=dict(zip(labels_,features_))
	
	return description

def get_descriptive_statistics(dict_, labels_):
	for j in range(len(labels_)):
		try:
			dict_[labels[j]]=str(np.mean(np.array(dict_[labels[j]])))+' (+/- '+str(np.std(np.array(dict_[labels[j]])))+')'
		except:
			dict_[labels[j]]='ERROR'
			
	return dict_

# go to the right folder
directory=sys.argv[1]
os.chdir(directory)
listdir=os.listdir()
jsonfiles=list()
for i in range(len(listdir)):
	if listdir[i].endswith('.json'):
		jsonfiles.append(listdir[i])

# got all the jsonfiles, now add to each feature 
print(jsonfiles)
description=describe_text(jsonfiles[0])
labels=list(description)

dict_=dict()
for i in range(len(labels)):
	dict_[labels[i]]=[]

# now go through all the json files 
for i in range(len(jsonfiles)):
	stats=describe_text(jsonfiles[i])
	print(stats)
	for j in range(len(labels)):
		try:
			dict_[labels[j]]=dict_[labels[j]]+[stats[labels[j]]]
		except:
			pass

dict_=get_descriptive_statistics(dict_, labels)

text_stats=dict_

# make the table! (alphabetized)
text_stats['A_Feature']='Average (+/- standard deviation)'
text_stats = dict(sorted(text_stats.items()))

# customizing the table
word_document = Document()
document_name = directory
table = word_document.add_table(0, 0) # we add rows iteratively
table.style = 'TableGrid'
first_column_width = 5
second_column_with = 10
table.add_column(Cm(first_column_width))
table.add_column(Cm(second_column_with))

for index, stat_item in enumerate(text_stats.items()):
	table.add_row()
	stat_name, stat_result = stat_item
	row = table.rows[index]
	row.cells[0].text = str(stat_name)
	row.cells[1].text = str(stat_result)
word_document.add_page_break()

word_document.save(directory + '.docx')
