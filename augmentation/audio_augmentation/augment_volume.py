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


 / _ \                                 | |      | | (_)            
/ /_\ \_   _  __ _ _ __ ___   ___ _ __ | |_ __ _| |_ _  ___  _ __  
|  _  | | | |/ _` | '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \ 
| | | | |_| | (_| | | | | | |  __/ | | | || (_| | |_| | (_) | | | |
\_| |_/\__,_|\__, |_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|
			  __/ |                                                
			 |___/                                                 
  ___  ______ _____        ___            _ _       
 / _ \ | ___ \_   _|  _   / _ \          | (_)      
/ /_\ \| |_/ / | |   (_) / /_\ \_   _  __| |_  ___  
|  _  ||  __/  | |       |  _  | | | |/ _` | |/ _ \ 
| | | || |    _| |_   _  | | | | |_| | (_| | | (_) |
\_| |_/\_|    \___/  (_) \_| |_/\__,_|\__,_|_|\___/   

takes in an audio file and outputs files normalized to 
different volumes. This corrects for microphone distance and ages.

Note that in using ffmpeg-normalize, this mimicks real world-use.
An alternative could be to use SoX to move up or down volume.
'''
import os

def augment_volume(filename):

	def change_volume(filename, vol):
		# rename file
		if vol > 1:
			new_file=filename[0:-4]+'_increase_'+str(vol)+'.wav'
		else:
			new_file=filename[0:-4]+'_decrease_'+str(vol)+'.wav'

		# changes volume, vol, by input 
		os.system('sox -v %s %s %s'%(str(vol),filename,new_file))

		return new_file 

	basefile=filename[0:-4]
	# using peak normalization
	os.system('ffmpeg-normalize %s -nt peak -t 0 -o %s_peak_normalized.wav'%(filename, basefile))
	
	filenames=list()
	# increase volume by 2x 
	new_file=change_volume(filename, 3)
	filenames.append(new_file)
	# decrease volume by 1/2 
	new_file=change_volume(filename, 0.33)
	filenames.append(new_file)
	return filenames