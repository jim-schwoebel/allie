'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: nnp_freq

#nnp = noun, proper, singular 
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
from collections import Counter

def nnp_freq(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)
    c=Counter(token for word, token in tokens)

    return c['NNP']/len(text)

