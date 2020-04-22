'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: to_freq

#to = 'to' as a preposition or infinitive marker
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
from collections import Counter

def to_freq(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)
    c=Counter(token for word, token in tokens)

    return ['TO']/len(text)

