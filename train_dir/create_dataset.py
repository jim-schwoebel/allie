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

Usage: python3 create_dataset.py [csvfile] [targetname]

Example: python3 create_dataset.py What_is_your_total_household_income.csv 'What is your total household income?'
--> outputs two folders: ['What is your total household income_below','What is your total household income_above']

Note that these two classes are centered around the mean of a regression problem spreadsheet (e.g. to allow for 
data visualization).
'''
import pandas as pd
import numpy as np
import os, shutil, sys

csvfile=sys.argv[1]
target=sys.argv[2]

data=pd.read_csv(csvfile)
urls=(data['url'])
data_values=list(data[target])
os.mkdir(target+'_above')
os.mkdir(target+'_below')
average=float(np.average(np.array(data_values)))

for i in range(len(data)):
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
