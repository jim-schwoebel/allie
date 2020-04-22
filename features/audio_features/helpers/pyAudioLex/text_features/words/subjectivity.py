'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: subjectivity

Take in a text sample and output the average, standard deviation, and variance
subjectivity. 
'''


import nltk
from nltk import word_tokenize
from textblob import TextBlob
import numpy as np

def subjectivity(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)

    #sentiment subjectivity of the session
    sentiment=TextBlob(importtext).sentiment[1]

    subjectivitylist=list()

    for i in range(0,len(tokens),3):
        if i <= len(tokens)-3:
            words=text[i]+' '+text[i+1]+' '+text[i+2]
            subjectivitylist.append(TextBlob(words).sentiment[1])
        else:
            pass
    
    #average difference subjectivity every 3 words
    avgsubjectivity=np.mean(subjectivitylist)

    #std subjectivity every 3 words
    stdsubjectivity=np.std(subjectivitylist)

    #var subjectivity every 3 words
    varsubjectivity=np.var(subjectivitylist)

    return [float(avgsubjectivity), float(stdsubjectivity), float(varsubjectivity)]


g=subjectivity('hello I suck so much right now')
print(g)
