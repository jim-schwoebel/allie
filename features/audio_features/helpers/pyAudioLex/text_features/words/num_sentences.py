'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: num_sentences

Total number of sentences, as calculated by punctuation marks:

Periods (.),
Interjections (!),
Questions (?).

'''

def num_sentences(importtext):

    #actual number of periods 
    periods=importtext.count('.')

    #count number of questions
    questions=importtext.count('?')

    #count number of interjections
    interjections=importtext.count('!')

    #actual number of sentences
    sentencenum=periods+questions+interjections 

    return [sentencenum,periods,questions,interjections]

#print(num_sentences('In a blunt warning to the remaining ISIS fighters, Army Command Sgt. Maj. John Wayne Troxell said the shrinking band of militants could either surrender to the U.S. military or face death.“ISIS needs to understand that the Joint Force is on orders to annihilate them,” he wrote in a forceful message on Facebook. “So they have two options, should they decide to come up against the United States, our allies and partners: surrender or die!'))
