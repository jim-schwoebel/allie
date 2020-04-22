'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: rb_freq

#rb = adverb
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
from collections import Counter

def rb_freq(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)
    c=Counter(token for word, token in tokens)

    return ['RB']/len(text)
