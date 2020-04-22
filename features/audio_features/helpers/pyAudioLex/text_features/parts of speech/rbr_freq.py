'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: rbr_freq

#rbr = adverb, comparative
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
from collections import Counter

def rbr_freq(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)
    c=Counter(token for word, token in tokens)

    return ['RBR']/len(text)
