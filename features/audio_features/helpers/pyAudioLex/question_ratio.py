'''
@package: pyAudioLex
@author: Drew Morris
@module: question_ratio

Patients are more likely to forget details in the middle of conversation, 
to not understand the questions, or to forget the context of the question. 
In those cases, they tend to ask the interviewer to repeat the question or 
they get confused, talk to themselves, and ask further questions about the 
details. The question words such as 'which,' 'what,' etc. are tagged 
automatically in each conversation. The full list of question tags that 
were used here is shown in Table 2. The question ratio of a subject is 
computed by dividing the total number of question words by the number 
of utterances spoken by the subject.
'''

from nltk.tokenize import RegexpTokenizer, word_tokenize

def question_ratio(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  tokenizer = RegexpTokenizer('Who|What|When|Where|Why|How|\?')
  qtokens = tokenizer.tokenize(s)

  if len(tokens) == 0:
    return float(0)
  else:
    return float(len(qtokens)) / float(len(tokens))