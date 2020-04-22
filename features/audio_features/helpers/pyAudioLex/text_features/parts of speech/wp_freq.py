'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: wrb_freq

#wrb = wh-adverb
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
from collections import Counter

def wrb_freq(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)
    c=Counter(token for word, token in tokens)

    return ['WRB']/len(text)
