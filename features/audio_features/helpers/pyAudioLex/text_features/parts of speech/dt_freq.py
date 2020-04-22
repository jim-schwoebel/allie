'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: dt_freq

#dt = determiner frequency

increased use of determiners a signal for schizophrenia.
'''

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
from collections import Counter

def dt_freq(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)
    c=Counter(token for word, token in tokens)

    return c['DT']/len(text)
