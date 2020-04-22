'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: freq_dist

outputs most frequent to least frequent words 
'''

from nltk.tokenize import word_tokenize
from nltk import FreqDist

def freq_dist(importtext):
    
    text=word_tokenize(importtext)
    fdist1=FreqDist(text)
    distribution=fdist1.most_common(len(fdist1))
    
    return distribution 

