'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: word_endings

Given a word ending (e.g. '-ed'), output the words with that ending and the associated count.
'''

from nltk import word_tokenize
import re 

def word_endings(importtext,ending):
    
    text=word_tokenize(importtext)
    
    #number of words ending in 'ed'
    words=[w for w in text if re.search(ending+'$', w)]
    
    return [len(words),words]

#test 
#print(word_endings('In a blunt warning to the remaining ISIS fighters, Army Command Sgt. Maj. John Wayne Troxell said the shrinking band of militants could either surrender to the U.S. military or face death. “ISIS needs to understand that the Joint Force is on orders to annihilate them,” he wrote in a forceful message on Facebook. “So they have two options, should they decide to come up against the United States, our allies and partners: surrender or die!”','s'))
