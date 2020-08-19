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

Make many CSV files for modeling,
good to combine with https://github.com/jim-schwoebel/allie/blob/master/training/regression_all.py

'''
import os
import pandas as pd

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

data=pd.read_csv('new3.csv')
cols=list(data)

for i in range(len(cols)-1):
	ind1=cols.index('url')
	ind2=cols.index(cols[i+1])
	delcols=list()
	for j in range(len(cols)):
		if j in [ind1,ind2]:
			pass
		else:
			delcols.append(cols[j])
	print(delcols)
	newdata=data.drop(delcols, axis=1)
	newstring=replace_nonstrings(cols[i+1])
	newdata.to_csv(newstring+'.csv', index=False)