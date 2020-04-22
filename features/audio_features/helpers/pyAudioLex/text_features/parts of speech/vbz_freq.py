'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: vbz_freq

#vbz = verb, present tense, 3rd person singular
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
from collections import Counter

def vbz_freq(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)
    c=Counter(token for word, token in tokens)

    return ['VBZ']/len(text)

