'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: nn_freq

#nn = noun, common, singular or mass
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
from collections import Counter

def nn_freq(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)
    c=Counter(token for word, token in tokens)

    return c['NN']/len(text)

