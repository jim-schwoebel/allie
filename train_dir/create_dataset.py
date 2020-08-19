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
			       
			       
How to use:

python3 create_dataset.py [csvfile] [targetname]

python3 create_dataset.py What_is_your_total_household_income.csv 'What is your total household income?'
'''
import pandas as pd
import numpy as np
import os, shutil, sys

def determine_categorical(data_values):
	numvalues=len(set(list(data_values)))
	uniquevals=list(set(list(data_values)))
	categorical=False
	if numvalues <= 10:
		categorical = True
	else:
		categorical = False
	return categorical, uniquevals

def replace_nonstrings(string_):
	# alphabet to keep characters
	alphabet=['a','b','c','d','e','f','g','h','i','j',
			  'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ','_',
			  '1','2','3','4','5','6','7','8','9','0']
	string_=string_.lower().replace(' ','_')
	newstring=''
	for j in range(len(string_)):
		if string_[j] not in alphabet:
			pass
		else:
			newstring=newstring+string_[j]

	if len(newstring) > 50:
		newstring=newstring[0:50]
		
	return newstring

csvfile=sys.argv[1]
target=sys.argv[2]

data=pd.read_csv(csvfile)
urls=(data['url'])
data_values=list(data[target])

average=float(np.average(np.array(data_values)))
categorical, uniquevals = determine_categorical(data_values)

target=replace_nonstrings(target)

if categorical == False:
	try:
		os.mkdir(target+'_above')
	except:
		shutil.rmtree(target+'_above')
	try:
		os.mkdir(target+'_below')
	except:
		shutil.rmtree(target+'_below')
else:
	for i in range(len(uniquevals)):
		newstring=replace_nonstrings(str(uniquevals[i]))
		try:
			os.mkdir(target+'_'+newstring)
		except:
			shutil.rmtree(target+'_'+newstring)

for i in range(len(data)):
	if categorical == False:
		if data_values[i] > average:
			shutil.copy(urls[i], os.getcwd()+'/'+target+'_above'+'/'+urls[i].split('/')[-1])
			try:
				shutil.copy(urls[i][0:-4]+'.json', os.getcwd()+'/'+target+'_above'+'/'+urls[i].split('/')[-1][0:-4]+'.json')
			except:
				pass
		else:
			shutil.copy(urls[i], os.getcwd()+'/'+target+'_below'+'/'+urls[i].split('/')[-1])
			try:
				shutil.copy(urls[i][0:-4]+'.json', os.getcwd()+'/'+target+'_below'+'/'+urls[i].split('/')[-1][0:-4]+'.json')
			except:
				pass
	else:
		for j in range(len(uniquevals)):
			if data_values[i] == uniquevals[j]:		
				newstring=replace_nonstrings(str(uniquevals[j]))
				shutil.copy(urls[i], os.getcwd()+'/'+target+'_'+newstring+'/'+urls[i].split('/')[-1])
				try:
					shutil.copy(urls[i][0:-4]+'.json', os.getcwd()+'/'+target+'_'+newstring+'/'+urls[i].split('/')[-1][0:-4]+'.json')
				except:
					pass
