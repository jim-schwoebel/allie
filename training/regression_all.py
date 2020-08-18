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

Builds many regression models based on .CSV files located in the train_dir.

Relies on this script: https://github.com/jim-schwoebel/allie/blob/master/train_dir/make_csv_regression.py

Note this is for single target regression problems only.
'''
import os, shutil
import pandas as pd

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

curdir=os.getcwd()
prevdir=prev_dir(curdir)
train_dir=prevdir+'/train_dir'
os.chdir(train_dir)
listdir=os.listdir()
csvfiles=list()
for i in range(len(listdir)):
	if listdir[i].endswith('.csv'):
		csvfiles.append(listdir[i])

# now train models
os.chdir(curdir)

for i in range(len(csvfiles)):
	os.chdir(train_dir)
	data=pd.read_csv(csvfiles[i])
	class_=list(data)[1]
	os.chdir(curdir)

	# make regression model
	os.system('python3 model.py r "%s" "%s"'%(csvfiles[i], class_))
	os.chdir(prevdir+'/train_dir/')
	
	# make classification model (allows for visualizations and averages around mean)
	os.system('python3 create_dataset.py "%s" "%s"'%(csvfiles[i], class_))
	os.chdir(curdir)
	os.system('python3 model.py audio 2 c "%s" "%s" "%s"'%(class_, class_+'_above', class_+'_below'))

	# remove temporary directories for classification model training
	try:
		shutil.rmtree(prevdir+'/train_dir/'+class_+'_above')
	except:
		pass
	try:
		shutil.rmtree(prevdir+'/train_dir/'+class_+'_below')
	except:
		pass