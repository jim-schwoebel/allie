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


|  ___|       | |                        / _ \ | ___ \_   _|  _ 
| |_ ___  __ _| |_ _   _ _ __ ___  ___  / /_\ \| |_/ / | |   (_)
|  _/ _ \/ _` | __| | | | '__/ _ \/ __| |  _  ||  __/  | |      
| ||  __/ (_| | |_| |_| | | |  __/\__ \ | | | || |    _| |_   _ 
\_| \___|\__,_|\__|\__,_|_|  \___||___/ \_| |_/\_|    \___/  (_)
                                                                
                                                                
  ___            _ _       
 / _ \          | (_)      
/ /_\ \_   _  __| |_  ___  
|  _  | | | |/ _` | |/ _ \ 
| | | | |_| | (_| | | (_) |
\_| |_/\__,_|\__,_|_|\___/ 
                           
This will featurize folders of audio files if the default_audio_features = ['gemaps_time_features']

This is the time series features for GeMAPS.

This is using OpenSMILE's new python library: https://github.com/audeering/opensmile-python
'''

import opensmile, json

def featurize_opensmile(wav_file):

	# initialize features and labels
	labels=list()
	features=list()

	# extract LLD 
	smile_LLD = opensmile.Smile(
	    feature_set=opensmile.FeatureSet.GeMAPSv01b,
	    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
	)

	y_LLD = smile_LLD.process_file(wav_file)

	labels_LLD=list(y_LLD)

	for i in range(len(labels_LLD)):
		features.append(list(y_LLD[labels_LLD[i]]))
		labels.append(labels_LLD[i])

	smile_LLD_deltas = opensmile.Smile(
	    feature_set=opensmile.FeatureSet.GeMAPSv01b,
	    feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,

	)

	y_LLD_deltas = smile_LLD_deltas.process_file(wav_file)

	labels_LLD_deltas=list(y_LLD_deltas)

	for i in range(len(labels_LLD_deltas)):
		features.append(list(y_LLD_deltas[labels_LLD_deltas[i]]))
		labels.append(labels_LLD_deltas[i])

	smile_functionals = opensmile.Smile(
	    feature_set=opensmile.FeatureSet.GeMAPSv01b,
	    feature_level=opensmile.FeatureLevel.Functionals,
	)

	y_functionals = smile_functionals.process_file(wav_file)

	labels_y_functionals=list(y_functionals)

	for i in range(len(labels_y_functionals)):
		features.append(list(y_functionals[labels_y_functionals[i]]))
		labels.append(labels_y_functionals[i])

	return features, labels
