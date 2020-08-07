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
  ___            _ _       
 / _ \          | (_)      
/ /_\ \_   _  __| |_  ___  
|  _  | | | |/ _` | |/ _ \ 
| | | | |_| | (_| | | (_) |
\_| |_/\__,_|\__,_|_|\___/ 
						   

This cleaning script converts a folder of .WAV audio files into .OPUS format, 
then converts this .OPUS file back to .WAV, replacing the original file.

.OPUS is a lossy codec and format that narrows in on the human voice range, 
so it could filter out other noises that are beyond the human voice range (20 Hz - 20 kHz).

This cleaning script is enabled if default_audio_cleaners=['clean_opus'] 
'''
import os, shutil

def clean_opus(filename, opusdir):

	filenames=list()

	#########################
	# lossy codec - .opus 
	#########################
	curdir=os.getcwd()
	newfile=filename[0:-4]+'.opus'

	# copy file to opus encoding folder 
	shutil.copy(curdir+'/'+filename, opusdir+'/'+filename)
	os.chdir(opusdir)
	print(os.getcwd())
	# encode with opus codec 
	os.system('opusenc %s %s'%(filename,newfile))
	os.remove(filename)
	filename=filename[0:-4]+'_opus.wav'
	os.system('opusdec %s %s'%(newfile, filename))
	os.remove(newfile)
	# delete .wav file in original dir 
	shutil.copy(opusdir+'/'+filename, curdir+'/'+filename)
	os.remove(filename)
	os.chdir(curdir)
	os.remove(newfile[0:-5]+'.wav')
	return [filename]