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
						   
Makes a table in microsoft word for all the audio features present in a file in a particular folder.

Usage: python3 get_stats.py [folder]

Example: python3 get_stats.py females
'''

from docx import Document
from docx.shared import Cm, Pt
import numpy as np
import os, json, time

def describe_text(jsonfile):

	# get dictionary 
	g=json.load(open(jsonfile))
	features=g['features']['audio']
	featuretypes=list(features)

	features_=list()
	labels_=list()
	for j in range(len(featuretypes)):
		features_=features_+features[featuretypes[j]]['features']
		labels_=labels_+features[featuretypes[j]]['labels']

	description=dict(zip(labels_,features_))
	
	return description

def get_descriptive_statistics(dict_, labels_):
	for j in range(len(labels_)):
		dict_[labels[j]]=str(np.mean(np.array(dict_[labels[j]])))+' (+/- '+str(np.std(np.array(dict_[labels[j]])))+')'
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
		dict_[labels[j]]=dict_[labels[j]]+[stats[labels[j]]]

dict_=get_descriptive_statistics(dict_, labels)

text_stats=dict_

# make the table! (alphabetized)
text_stats['A_Title']='TITLE'
text_stats = dict(sorted(text_stats.items()))

# customizing the table
word_document = Document()
document_name = 'news-article-stats'
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

word_document.save(document_name + '.docx')
