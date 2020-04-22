'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: word_repeats

Word repeats on average every 10 words (important for psychosis).

Typical repeated word output:

the
the
the
the
to
to
to
to
on
on
,
,

If words are typically not articles like 'the', it could indicate a thought disorder. 
'''

from nltk import word_tokenize

def word_repeats(importtext):
    
    tokens=word_tokenize(importtext)
    
    tenwords=list()
    tenwords2=list()
    repeatnum=0
    repeatedwords=list()

    #make number of sentences 
    for i in  range(0,len(tokens),10):
        tenwords.append(i)
        
    for j in range(0,len(tenwords)):
        if j not in [len(tenwords)-2,len(tenwords)-1]:
            tenwords2.append(tokens[tenwords[j]:tenwords[j+1]])
        else:
            pass

    #now parse for word repeats sentence-over-sentence 
    for k in range(0,len(tenwords2)):
        if k<len(tenwords2)-1:
            for l in range(10):
                if tenwords2[k][l] in tenwords2[k+1]:
                    repeatnum=repeatnum+1
                    repeatedwords.append(tenwords2[k][l])
                if tenwords2[k+1][l] in tenwords2[k]:
                    repeatnum=repeatnum+1
                    repeatedwords.append(tenwords2[k+1][l])
        else:
            pass

    print

    #calculate the number of sentences and repeat word avg per sentence 
    sentencenum=len(tenwords)
    repeatavg=repeatnum/sentencenum

    #repeated word freqdist 

    return [repeatedwords, sentencenum, repeatavg]

#test 
#print(word_repeats('In a blunt warning to the remaining ISIS fighters, Army Command Sgt. Maj. John Wayne Troxell said the shrinking band of militants could either surrender to the U.S. military or face death. “ISIS needs to understand that the Joint Force is on orders to annihilate them,” he wrote in a forceful message on Facebook. “So they have two options, should they decide to come up against the United States, our allies and partners: surrender or die!”'))
