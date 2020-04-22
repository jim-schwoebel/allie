'''
@package: pyAudioLex
@author: Drew Morris
@module: number_ratio

During conversations, subjects give details about their birth dates, 
how many kids they have, and other numerical information. Such use of 
numbers in a sentence can be a measure of recall ability. The number ratio 
feature is calculated by dividing the total count of numbers by the total 
count of words the subject used in the conversation.
'''

from nltk.tokenize import RegexpTokenizer, word_tokenize

def number_ratio(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)

  tokenizer = RegexpTokenizer('zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|dozen|couple|several|few|\d')
  qtokens = tokenizer.tokenize(s.lower())

  if len(tokens) == 0:
    return float(0)
  else:
    return float(len(qtokens)) / float(len(tokens))
