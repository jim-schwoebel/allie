'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: datwords_freq

Take in a text sample and output frequency of date-related words.
'''

import nltk
from nltk import word_tokenize

def datewords_freq(importtext):

    text=word_tokenize(importtext)
    datewords=['time','monday','tuesday','wednesday','thursday','friday','saturday','sunday','january','february','march','april','may','june','july','august','september','november','december','year','day','hour','today','month',"o'clock","pm","am"]
    datewords2=list()
    for i in range(len(datewords)):
        datewords2.append(datewords[i]+'s')
    
    datewords=datewords+datewords2
    print(datewords)                      
    datecount=0

    for i in range(len(text)):
        if text[i].lower() in datewords:
            datecount=datecount+1
            
    datewords=datecount

    datewordfreq=datecount/len(text)

    return datewordfreq

print(datewords_freq('I love you jess on vonly'))
