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
 _____         _   
|_   _|       | |  
	| | _____  _| |_ 
	| |/ _ \ \/ / __|
	| |  __/>  <| |_ 
	\_/\___/_/\_\\__|
									 
This script takes a folder of .TXT files and extracts a 100 word summary from 
each and replaces the original text file with the summary. This is useful if you
have a lot of very long text files that you want to analyze for machine learning 
purposes (e.g. books).

This is enabled if the default_text_cleaners=['clean_summary']
'''
import os, uuid, sys

try:
	import textrank
except:
	os.system('pip3 install git+git://github.com/davidadamojr/TextRank.git')

def clean_summary(textfile):
	summary='summary_'+textfile
	os.system('textrank extract-summary %s > %s'%(textfile, summary))
	summary=open(summary).read()
	os.remove(textfile)
	return [summary]