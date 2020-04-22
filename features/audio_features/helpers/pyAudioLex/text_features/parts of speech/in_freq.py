'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: in_freq

in = preposition or conjunction, suborbinating
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
from collections import Counter

def in_freq(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)
    c=Counter(token for word, token in tokens)

    return c['IN']/len(text)


