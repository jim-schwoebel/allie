'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: polarity

Take in a text sample and output the average, standard deviation, and variance
polarity. (+) indicates happy and (-) indicates sad, 0 is neutral. 
'''

import nltk
from nltk import word_tokenize
from textblob import TextBlob
import numpy as np 

def polarity(importtext):
    
    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)

    #sentiment polarity of the session
    polarity=TextBlob(importtext).sentiment[0]

    #sentiment subjectivity of the session
    sentiment=TextBlob(importtext).sentiment[1]

    #average difference polarity every 3 words
    polaritylist=list()
    for i in range(0,len(tokens),3):
        if i <= len(tokens)-3:
            words=text[i]+' '+text[i+1]+' '+text[i+2]
            polaritylist.append(TextBlob(words).sentiment[0])
        else:
            pass 
    avgpolarity=np.mean(polaritylist)

    #std polarity every 3 words
    stdpolarity=np.std(polaritylist)

    #variance polarity every 3 words
    varpolarity=np.var(polaritylist)

    return [float(avgpolarity), float(stdpolarity), float(varpolarity)]

