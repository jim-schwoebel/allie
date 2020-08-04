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
			       
Renames all the files in a particular directory (both audio files and .JSON files). 

Note you can manually change this to other file types.

Usage: python3 rename_files.py [folder]
Example: python3 rename.py /Users/jim/desktop/allie/train_dir/males
'''
import os,uuid, sys

directory=sys.argv[1]
os.chdir(directory)
listdir=os.listdir()

for i in range(len(listdir)):
	if listdir[i].endswith('.wav'):
		newname=str(uuid.uuid4())
		os.rename(listdir[i],newname+'.wav')
		if listdir[i][0:-4]+'.json' in listdir:
			os.rename(listdir[i][0:-4]+'.json', newname+'.json')
