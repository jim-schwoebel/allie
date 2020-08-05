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

______         _                          ___  ______ _____     
|  ___|       | |                        / _ \ | ___ \_   _|  
| |_ ___  __ _| |_ _   _ _ __ ___  ___  / /_\ \| |_/ / | |   
|  _/ _ \/ _` | __| | | | '__/ _ \/ __| |  _  ||  __/  | |      
| ||  __/ (_| | |_| |_| | | |  __/\__ \ | | | || |    _| |_   
\_| \___|\__,_|\__|\__,_|_|  \___||___/ \_| |_/\_|    \___/  
                                                                
                                     
This is the standard feature array for Allie (version 1.0).

Note this will be imported to get back data in all featurization methods
to ensure maximal code reusability.
'''
import os, time, psutil, json, platform
from datetime import datetime
		
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

def make_features(sampletype):

	# only add labels when we have actual labels.
	features={'audio':dict(),
		      'text': dict(),
		      'image':dict(),
		      'video':dict(),
		      'csv': dict()}

	transcripts={'audio': dict(),
				 'text': dict(),
				 'image': dict(),
				 'video': dict(),
				 'csv': dict()}
			   
	models={'audio': dict(),
			'text': dict(),
			'image': dict(),
			'video': dict(),
			'csv': dict()}
	
	# getting settings can be useful to see if settings are the same in every
	# featurization, as some featurizations can rely on certain settings to be consistent
	prevdir=prev_dir(os.getcwd())
	try:
		settings=json.load(open(prevdir+'/settings.json'))
	except:
		# this is for folders that may be 2 layers deep in train_dir
		settings=json.load(open(prev_dir(prevdir)+'/settings.json'))
	
	data={'sampletype': sampletype,
		  'transcripts': transcripts,
		  'features': features,
		  'models': models,
		  'labels': [],
		  'errors': [],
		  'settings': settings,
		 }
	
	return data
