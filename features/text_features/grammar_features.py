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
                   
		   
Featurize folders of text files if default_text_features = ['grammar_features']

Inputs a text file and featurizes the text into many grammatical features. 
This will produce a sparse matrix with many zeros, with a few significant features
and is memory-intensive.
'''
from itertools import permutations
import nltk 
from nltk import load, word_tokenize

def grammar_featurize(importtext):
    
    #now have super long string of text can do operations 
    #get all POS fromo Penn Treebank (POS tagger)
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    nltk_pos_list=tagdict.keys()
    #get all permutations of this list 
    perm=permutations(nltk_pos_list,3)
    #make these permutations in a list 
    listobj=list()
    for i in list(perm):
        listobj.append(list(i))

    #split by sentences? or by word? (will do by word here)
    text=word_tokenize(importtext)
    #tokens now 
    tokens=nltk.pos_tag(text)
    #initialize new list for pos
    pos_list=list()

    #parse through entire document and tokenize every 3 words until end 
    for i in range(len(tokens)-3):
        pos=[tokens[i][1],tokens[i+1][1],tokens[i+2][1]]
        pos_list.append(pos)

    #count each part of speech event and total event count 
    counts=list()
    totalcounts=0
    for i in range(len(listobj)):
        count=pos_list.count(listobj[i])
        totalcounts=totalcounts+count 
        counts.append(count)

    #now create probabilities / frequencies from total count
    freqs=list()
    for i in range(len(counts)):
        freq=counts[i]/totalcounts
        freqs.append(freq)

    #now you can sort lowest to highest frequency (this is commented out to keep order consistent) 
    # listobj.sort(key=lambda x: (x[3]),reverse=True)
    features=freqs
    labels=listobj

    return features, labels
