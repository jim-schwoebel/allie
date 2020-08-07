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

Following remove_noise.py from voicebook.
'''
import os
import soundfile as sf

def augment_noise(filename):
	#now use sox to denoise using the noise profile
	data, samplerate =sf.read(filename)
	duration=data/samplerate
	first_data=samplerate/10
	filter_data=list()
	for i in range(int(first_data)):
		filter_data.append(data[i])
	noisefile='noiseprof.wav'
	sf.write(noisefile, filter_data, samplerate)
	os.system('sox %s -n noiseprof noise.prof'%(noisefile))
	filename2='tempfile.wav'
	filename3='tempfile2.wav'
	noisereduction="sox %s %s noisered noise.prof 0.21 "%(filename,filename2)
	command=noisereduction
	#run command 
	os.system(command)
	print(command)
	#rename and remove files 
	newfile=filename[0:-4]+'_noise_remove.wav'
	os.rename(filename2,newfile)
	#os.remove(filename2)
	os.remove(noisefile)
	os.remove('noise.prof')
	return [newfile]
