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

|  \/  |         | |    | |    
| .  . | ___   __| | ___| |___ 
| |\/| |/ _ \ / _` |/ _ \ / __|
| |  | | (_) | (_| |  __/ \__ \
\_|  |_/\___/ \__,_|\___|_|___/

Can be useful to count the number of model predictions in a certain class after prediction 
is complete to get an intuition of accuracy of model on new datasets. 

Note you currently have to manually edit this file for it to be useful.

Usage: python3 validate.py
'''
import os,json
from tqdm import tqdm
import time
def prev_dir(directory):
	g=directory.split('/')
	dir_=''
	for i in range(len(g)):
		if i != len(g)-1:
			if i==0:
				dir_=dir_+g[i]
			else:
				dir_=dir_+'/'+g[i]
	# print(dir_)
	return dir_

os.chdir(prev_dir(os.getcwd())+'/load_dir/')
listdir=os.listdir()
jsonfiles=list()

for i in range(len(listdir)):
    if listdir[i].endswith('.json'):
            jsonfiles.append(listdir[i])

predictions=list()
for i in tqdm(range(len(jsonfiles))):
	try:
		g=json.load(open(jsonfiles[i]))
		models=list(g['models']['audio'])
		predictions=predictions+models
	except:
		print('error')

print('females')
print(predictions.count('female'))
print('males')
print(predictions.count('male'))
