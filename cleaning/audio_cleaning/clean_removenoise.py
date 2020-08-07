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
						   												 

This cleaning script de-noises all audio files in a given folder using a SoX noise profile. 
This is done by taking the first 500 milliseconds and using this as a basis to delete noise 
out of the rest of the file. Note that this works well if the noise is linear but not well
if the noise is non-linear.

This cleaning script is enabled if default_audio_cleaners=['clean_removenoise'] 
'''
import os, uuid

def clean_removenoise(audiofile):
	# create a noise reference (assuming linear noise)
	# following https://stackoverflow.com/questions/44159621/how-to-denoise-audio-with-sox
	# alternatives would be to use bandpass filter or other low/hf filtering techniques
	noiseaudio=str(uuid.uuid1())+'_noiseaudio.wav'
	noiseprofile=str(uuid.uuid1())+'_noise.prof'
	temp=audiofile[0:-4]+'_.wav'
	os.system('sox %s %s trim 0 0.500'%(audiofile, noiseaudio))
	os.system('sox %s -n noiseprof %s'%(noiseaudio, noiseprofile))
	os.system('sox %s %s noisered %s 0.21'%(audiofile, temp, noiseprofile))
	os.remove(audiofile)
	os.rename(temp,audiofile)
	os.remove(noiseaudio)
	os.remove(noiseprofile)
	return [audiofile]