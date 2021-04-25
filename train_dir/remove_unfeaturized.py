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
						   
Deletes all .JSON files from all folders in the train_dir 
(useful to re-featurize sets of files).

Usage: python3 delete_json.py

Example: python3 delete_json.py
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
		count=0
		for j in range(len(listdir)):
				if listdir[j].endswith('.wav') and listdir[j][0:-4]+'.json' not in listdir:
					print(folders[i])
					print(listdir[j])
					os.remove(listdir[j])
					count=count+1
					print(count)
