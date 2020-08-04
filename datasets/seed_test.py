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

______      _                 _              ___            _ _       
|  _  \    | |               | |        _   / _ \          | (_)      
| | | |__ _| |_ __ _ ___  ___| |_ ___  (_) / /_\ \_   _  __| |_  ___  
| | | / _` | __/ _` / __|/ _ \ __/ __|     |  _  | | | |/ _` | |/ _ \ 
| |/ / (_| | || (_| \__ \  __/ |_\__ \  _  | | | | |_| | (_| | | (_) |
|___/ \__,_|\__\__,_|___/\___|\__|___/ (_) \_| |_/\__,_|\__,_|_|\___/ 
                                                                    
Quickly generate some sample audio data from a GitHub repository.
'''

import os, shutil

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

listdir=os.listdir()
if 'sample_voice_data' not in listdir:
	os.system('git clone git@github.com:jim-schwoebel/sample_voice_data.git')
else:
	pass

cur_dir=os.getcwd()
base_dir=prev_dir(cur_dir)
train_dir=base_dir+'/train_dir'

try:
	shutil.copy(cur_dir+'/sample_voice_data/gender_all.csv',train_dir+'/gender_all.csv')
except:
	os.remove(train_dir+'/gender_all.csv')
	shutil.copy(cur_dir+'/sample_voice_data/gender_all.csv',train_dir+'/gender_all.csv')
try:
	shutil.copytree(cur_dir+'/sample_voice_data/males',train_dir+'/males')
except:
	shutil.rmtree(train_dir+'/males')
	shutil.copytree(cur_dir+'/sample_voice_data/males',train_dir+'/males')
try:
	shutil.copytree(cur_dir+'/sample_voice_data/females',train_dir+'/females')
except:
	shutil.rmtree(train_dir+'/females')
	shutil.copytree(cur_dir+'/sample_voice_data/females',train_dir+'/females')
