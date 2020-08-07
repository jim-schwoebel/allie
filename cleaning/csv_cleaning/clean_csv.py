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


/  __ \ |                (_)              / _ \ | ___ \_   _|  _ 
| /  \/ | ___  __ _ _ __  _ _ __   __ _  / /_\ \| |_/ / | |   (_)
| |   | |/ _ \/ _` | '_ \| | '_ \ / _` | |  _  ||  __/  | |      
| \__/\ |  __/ (_| | | | | | | | | (_| | | | | || |    _| |_   _ 
 \____/_|\___|\__,_|_| |_|_|_| |_|\__, | \_| |_/\_|    \___/  (_)
																	 __/ |                         
																	|___/                          
 _____  _____  _   _ 
/  __ \/  ___|| | | |
| /  \/\ `--. | | | |
| |     `--. \| | | |
| \__/\/\__/ /\ \_/ /
 \____/\____/  \___/ 

This section of Allie's API cleans folders of csv files
using the default_csv_cleaners.
'''
import os
import pandas as pd
try:
	import datacleaner
except:
	os.system('pip3 install datacleaner==0.1.5')

def clean_csv(csvfile, basedir):
	'''
	https://github.com/rhiever/datacleaner
	'''
	input_dataframe=pd.read_csv(csvfile)
	newframe=datacleaner.autoclean(input_dataframe, drop_nans=False, copy=False, ignore_update_check=False)
	newfile='clean_'+csvfile
	newframe.to_csv(newfile, index=False)
	return newfile