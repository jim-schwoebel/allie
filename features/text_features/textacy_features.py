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
|  ___|       | |                        / _ \ | ___ \_   _|  _ 
| |_ ___  __ _| |_ _   _ _ __ ___  ___  / /_\ \| |_/ / | |   (_)
|  _/ _ \/ _` | __| | | | '__/ _ \/ __| |  _  ||  __/  | |      
| ||  __/ (_| | |_| |_| | | |  __/\__ \ | | | || |    _| |_   _ 
\_| \___|\__,_|\__|\__,_|_|  \___||___/ \_| |_/\_|    \___/  (_)
                                                                
                                                                
 _____         _   
|_   _|       | |  
  | | _____  _| |_ 
  | |/ _ \ \/ / __|
  | |  __/>  <| |_ 
  \_/\___/_/\_\\__|
                   
		   
Featurize folders of text files if default_text_features = ['textacy_features']

Featurizes .TXT files with textacy - mostly intelligibility measures, as shown below.
More info on textacy can be found @ https://github.com/chartbeat-labs/textacy

-----

ts = textacy.TextStats(doc)

ts.n_unique_words
57
>>> ts.basic_counts
{'n_sents': 3,
 'n_words': 73,
 'n_chars': 414,
 'n_syllables': 134,
 'n_unique_words': 57,
 'n_long_words': 30,
 'n_monosyllable_words': 38,
 'n_polysyllable_words': 19}
>>> ts.flesch_kincaid_grade_level
15.56027397260274
>>> ts.readability_stats
{'flesch_kincaid_grade_level': 15.56027397260274,
 'flesch_reading_ease': 26.84351598173518,
 'smog_index': 17.5058628484301,
 'gunning_fog_index': 20.144292237442922,
 'coleman_liau_index': 16.32928468493151,
 'automated_readability_index': 17.448173515981736,
 'lix': 65.42922374429223,
 'gulpease_index': 44.61643835616438,
 'wiener_sachtextformel': 11.857779908675797}
'''
import textacy, os
import numpy as np 

def stats(matrix):
    mean=np.mean(matrix)
    std=np.std(matrix)
    maxv=np.amax(matrix)
    minv=np.amin(matrix)
    median=np.median(matrix)
    output=np.array([mean,std,maxv,minv,median])
    return output

def textacy_featurize(transcript):
    features=list()
    labels=list()

    # use Spacy doc
    try:
        doc = textacy.make_spacy_doc(transcript)
    except:
        os.system('python3 -m spacy download en')
        doc = textacy.make_spacy_doc(transcript)
    
    ts = textacy.TextStats(doc)
    uniquewords=ts.n_unique_words
    features.append(uniquewords)
    labels.append('uniquewords')

    mfeatures=ts.basic_counts
    features=features+list(mfeatures.values())
    labels=labels+list(mfeatures)

    kincaid=ts.flesch_kincaid_grade_level
    features.append(kincaid)
    labels.append('flesch_kincaid_grade_level')

    readability=ts.readability_stats
    features=features+list(readability.values())
    labels=labels+list(readability)
    
    return features, labels
