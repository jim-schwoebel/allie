'''
@package: pyAudioLex
@author: Drew Morris
@module: filler_ratio

Filler sounds such as 'ahm' and 'ehm' are used by people in spoken language 
when they think about what to say next. We hypothesize that they may be used 
more frequently by the patients because of slow thinking and memory recall 
processes. Patients tend to forget what they are talking about and to use 
fillers more often than the control subjects. The filler ratio is computed 
by dividing the total number of filler words by the total number of 
utterances spoken by the subject.
'''

from nltk.tokenize import RegexpTokenizer, word_tokenize

def filler_ratio(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  tokenizer = RegexpTokenizer('uh|ugh|um|like|you know')
  qtokens = tokenizer.tokenize(s.lower())

  if len(tokens) == 0:
    return float(0)
  else:
    return float(len(qtokens)) / float(len(tokens))
