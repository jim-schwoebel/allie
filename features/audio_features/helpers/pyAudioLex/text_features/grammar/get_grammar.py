'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: get_grammar

This script takes in a text sample and outputs 85140 grammar features.

Specifically, it extracts grammar from permutations of various parts of
speech in series.

For example, for the sentence 'I ate ham' it would be saved as
[Pronoun, Verb, Noun] for the first position.

The output is calculated in terms of the frequencies of these parts of speech,
from highest probability to lowest probability. 

This is important for many applications, as grammar is context-free and
often reflects the state-of-mind of the speaker.

'''

from itertools import permutations
import nltk 
from nltk import load, word_tokenize

def get_grammar(importtext):
    
    #now have super long string of text can do operations 
    #get all POS fromo Penn Treebank (POS tagger)
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    nltk_pos_list=tagdict.keys()
    #get all permutations of this list 
    perm=permutations(nltk_pos_list,3)
    #make these permutations in a list 
    listobj=list()
    for i in list(perm):
        listobj.append(list(i))

    #split by sentences? or by word? (will do by word here)
    text=word_tokenize(importtext)
    #tokens now 
    tokens=nltk.pos_tag(text)
    #initialize new list for pos
    pos_list=list()

    #parse through entire document and tokenize every 3 words until end 
    for i in range(len(tokens)-3):
        pos=[tokens[i][1],tokens[i+1][1],tokens[i+2][1]]
        pos_list.append(pos)

    #count each part of speech event and total event count 
    counts=list()
    totalcounts=0
    for i in range(len(listobj)):
        count=pos_list.count(listobj[i])
        totalcounts=totalcounts+count 
        counts.append(count)

    #now create probabilities / frequencies from total count
    freqs=list()
    for i in range(len(counts)):
        freq=counts[i]/totalcounts
        freqs.append(freq)

    #now you can append all the permutation labels with freqs
    for i in range(len(listobj)):
        listobj[i].append(freqs[i])

    #now you can sort lowest to highest frequency (this is commented out to keep order consistent) 
    # listobj.sort(key=lambda x: (x[3]),reverse=True)

    return listobj

