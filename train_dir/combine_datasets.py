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

python3 combine_datasests.py

Must edit the dir_1, dir_2, dir_1_folders, and dir_2_folders 

where:

dir_1 = first directory to combine all datasets in class 1 (e.g. male)
dir_2 = second directory to combine all datasets into class 2 (e.g. female)
dir_1_folders = a list of all the folders in the current directory tied to class 1 (to combine - e.g. male)
dir_2_folders = a list of all the folders in the current directory tied to class 2 (to combine - e.g. female)
'''
import os, shutil
from tqdm import tqdm

def copy_files(directory_):
	listdir=os.listdir()
	newdir=os.getcwd()
	for i in tqdm(range(len(listdir)), desc=newdir):
		if listdir[i].endswith('.json') or listdir[i].endswith('.wav'):
			shutil.copy(os.getcwd()+'/'+listdir[i], directory_+'/'+listdir[i])

curdir=os.getcwd()
dir_1=curdir+'/directory1_combined'
dir_2=curdir+'/directory2_combined'

# folders to combine into directory 1
dir_1_folders=['directory1_dataset1', 'directory1_dataset2']
for i in range(len(dir_1_folders)):
	os.chdir(curdir)
	os.chdir(dir_1_folders[i])
	copy_files(dir_1)

# folders to combine into directory 2
dir_2_folders=['directory2_dataset1', 'directory2_dataset2']
for i in range(len(dir_2_folders)):
	os.chdir(curdir)
	os.chdir(dir_2_folders[i])
	copy_files(dir_2)
