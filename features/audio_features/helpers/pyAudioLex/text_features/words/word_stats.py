'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: word_stats

Take in a text sample and output the average word length,
the maximum word length, the minimum word length,
the variance of the vocabulary, and the standard deviation of the vocabulary.

All of this is done by counting the length of individual words from tokens. 
'''

from nltk import word_tokenize
import numpy as np 

def word_stats(importtext):
    
    text=word_tokenize(importtext)

    #average word length 
    awords=list()
    for i in range(len(text)):
        awords.append(len(text[i]))
        
    awordlength=np.mean(awords)
    
    #all words greater than 5 in length
    fivewords= [w for w in text if len(w) > 5]
    fivewordnum=len(fivewords)

    #maximum word length
    vmax=np.amax(awords)

    #minimum word length
    vmin=np.amin(awords)

    #variance of the vocabulary
    vvar=np.var(awords)

    #stdev of vocabulary
    vstd=np.std(awords)

    return [float(awordlength),float(fivewordnum), float(vmax),float(vmin),float(vvar),float(vstd)]

#print(word_stats('hello this is test of the script.'))
