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

Randomly cut an audio file into a segment of 1 to 5 seconds.
'''
import os, random
import soundfile as sf 

def augment_randomsplice(filename):
	slicenum=random.randint(1,5)
	file=filename
	data, samplerate = sf.read(file)
	totalframes=len(data)
	totalseconds=int(totalframes/samplerate)
	startsec=random.randint(0,totalseconds-(slicenum+1))
	endsec=startsec+slicenum
	startframe=samplerate*startsec
	endframe=samplerate*endsec
	newfile='snipped%s_'%(str(slicenum))+file
	sf.write(newfile, data[int(startframe):int(endframe)], samplerate)
	return [newfile]
