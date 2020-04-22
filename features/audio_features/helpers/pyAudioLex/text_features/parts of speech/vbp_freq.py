'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: vbp_freq

#vbp = verb, present tense, not 3rd person singular
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
from collections import Counter

def vbp_freq(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)
    c=Counter(token for word, token in tokens)

    return ['VBP']/len(text)

