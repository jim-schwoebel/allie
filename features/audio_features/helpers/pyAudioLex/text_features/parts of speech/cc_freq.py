'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: cc_freq

In grammar, a conjunction (abbreviated conj or cnj) is a part of speech
that connects words, phrases, or clauses that are called the conjuncts
of the conjoining construction. The term discourse marker is mostly used
for conjunctions joining sentences. This definition may overlap with that
of other parts of speech, so what constitutes a "conjunction" must be defined
for each language. In general, a conjunction is an invariable grammatical
particle and it may or may not stand between the items in a conjunction.

Coordinating conjunctions, also called coordinators, are conjunctions that join,
or coordinate, two or more items (such as words, main clauses, or sentences) of
equal syntactic importance. In English, the mnemonic acronym FANBOYS can be used
to remember the coordinators for, and, nor, but, or, yet, and so.[4] These are
not the only coordinating conjunctions; various others are used, including[5]:
ch. 9[6]:p. 171 "and nor" (British), "but nor" (British), "or nor" (British),
"neither" ("They don't gamble; neither do they smoke"), "no more"
("They don't gamble; no more do they smoke"), and "only"
("I would go, only I don't have time"). Types of coordinating conjunctions
include cumulative conjunctions, adversative conjunctions, alternative conjunctions,
and illative conjunctions.[7]

Here are some examples of coordinating conjunctions in English and what they do:

For – presents rationale ("They do not gamble or smoke, for they are ascetics.")
And – presents non-contrasting item(s) or idea(s) ("They gamble, and they smoke.")
Nor – presents a non-contrasting negative idea ("They do not gamble, nor do they smoke.")
But – presents a contrast or exception ("They gamble, but they don't smoke.")
Or – presents an alternative item or idea ("Every day they gamble, or they smoke.")
Yet – presents a contrast or exception ("They gamble, yet they don't smoke.")
So – presents a consequence ("He gambled well last night, so he smoked a cigar to celebrate.")

'''

#POS not listed in the doc


from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
from collections import Counter

def cc_freq(importtext):

    text=word_tokenize(importtext)
    tokens=nltk.pos_tag(text)
    c=Counter(token for word, token in tokens)

    return c['CC']/len(text)
