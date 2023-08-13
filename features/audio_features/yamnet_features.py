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
                           

This will featurize folders of audio files if the default_audio_features = ['pspeech_features']

Python Speech Features is a library for fast extraction of speech features like mfcc coefficients and 
log filter bank energies. Note that this library is much faster than LibROSA and other libraries, 
so it is useful to featurize very large datasets.

For more information, check out the documentation: https://github.com/jameslyons/python_speech_features
'''
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import io, os, shutil, csv, pyaudio, wave
import soundfile as sf
from tqdm import tqdm
'''
https://tfhub.dev/google/yamnet/1
'''
# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:] #Skip CSV header
    return class_names

# get labels for later 
def get_labels(vector, label, label2):
    sample_list=list()
    for i in range(len(vector)):
        sample_list.append(label+str(i+1)+'_'+label2)

    return sample_list

def yamnet_featurize(wavfile, help_dir):
    model = hub.load(help_dir+'/yamnet_1')
    file_path = wavfile
    audio_data, sample_rate = sf.read(file_path)
    waveform = audio_data
    # Run the model, check the output.
    scores, embeddings, log_mel_spectrogram = model(waveform)
    scores.shape.assert_is_compatible_with([None, 521])
    embeddings.shape.assert_is_compatible_with([None, 1024])
    log_mel_spectrogram.shape.assert_is_compatible_with([None, 64])
    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
    features_mean = scores.numpy().mean(axis=0)
    features_std = scores.numpy().std(axis=0)
    features_max = scores.numpy().max(axis=0)
    features_min = scores.numpy().min(axis=0)
    features_median = np.median(scores.numpy(), axis=0)
    features=np.concatenate((features_mean, features_std, features_max, features_min, features_median), axis=0, out=None, dtype=None, casting="same_kind")

    labels=[]
    for i in range(len(class_names)):
        labels.append(class_names[i]+'_mean')
    for i in range(len(class_names)):
        labels.append(class_names[i]+'_std')
    for i in range(len(class_names)):
        labels.append(class_names[i]+'_max')
    for i in range(len(class_names)):
        labels.append(class_names[i]+'_min')
    for i in range(len(class_names)):
        labels.append(class_names[i]+'_medians')

    return features, labels

