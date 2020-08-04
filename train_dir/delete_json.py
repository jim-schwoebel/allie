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

|_   _|       (_)     (_)              / _ \ | ___ \_   _|
  | |_ __ __ _ _ _ __  _ _ __   __ _  / /_\ \| |_/ / | |  
  | | '__/ _` | | '_ \| | '_ \ / _` | |  _  ||  __/  | |  
  | | | | (_| | | | | | | | | | (_| | | | | || |    _| |_ 
  \_/_|  \__,_|_|_| |_|_|_| |_|\__, | \_| |_/\_|    \___/ 
                                __/ |                     
                               |___/       
			     	       
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
        for i in range(len(listdir)):
                if listdir[i].endswith('.json'):
                        print(listdir[i])
                        os.remove(listdir[i])
