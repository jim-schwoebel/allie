'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: vbn_freq

#vbn = verb, past participle
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
from collections import Counter

def vbn_freq(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)
    c=Counter(token for word, token in tokens)

    return ['VBN']/len(text)

