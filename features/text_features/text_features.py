'''
               AAA               lllllll lllllll   iiii                      
              A:::A              l:::::l l:::::l  i::::i                     
             A:::::A             l:::::l l:::::l   iiii                      
            A:::::::A            l:::::l l:::::l                             
           A:::::::::A            l::::l  l::::l iiiiiii     eeeeeeeeeeee    
          A:::::A:::::A           l::::l  l::::l i:::::i   ee::::::::::::ee  
         A:::::A A:::::A          l::::l  l::::l  i::::i  e::::::eeeee:::::ee
        A:::::A   A:::::A         l::::l  l::::l  i::::i e::::::e     e:::::e
       A:::::A     A:::::A        l::::l  l::::l  i::::i e:::::::eeeee::::::e
      A:::::AAAAAAAAA:::::A       l::::l  l::::l  i::::i e:::::::::::::::::e 
     A:::::::::::::::::::::A      l::::l  l::::l  i::::i e::::::eeeeeeeeeee  
    A:::::AAAAAAAAAAAAA:::::A     l::::l  l::::l  i::::i e:::::::e           
   A:::::A             A:::::A   l::::::ll::::::li::::::ie::::::::e          
  A:::::A               A:::::A  l::::::ll::::::li::::::i e::::::::eeeeeeee  
 A:::::A                 A:::::A l::::::ll::::::li::::::i  ee:::::::::::::e  
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee  

______         _                          ___  ______ _____     
|  ___|       | |                        / _ \ | ___ \_   _|  _ 
| |_ ___  __ _| |_ _   _ _ __ ___  ___  / /_\ \| |_/ / | |   (_)
|  _/ _ \/ _` | __| | | | '__/ _ \/ __| |  _  ||  __/  | |      
| ||  __/ (_| | |_| |_| | | |  __/\__ \ | | | || |    _| |_   _ 
\_| \___|\__,_|\__|\__,_|_|  \___||___/ \_| |_/\_|    \___/  (_)
                                                                
                                                                
 _____         _   
|_   _|       | |  
  | | _____  _| |_ 
  | |/ _ \ \/ / __|
  | |  __/>  <| |_ 
  \_/\___/_/\_\\__|
                   
		   
Featurize folders of text files if default_text_features = ['text_features']

This extracts many linguistic features such as the filler ratio,
type_token_ratio, entropy, standardized_word_entropy,
question_ratio, number_ratio, brunet's index, honore's statistic,
and many others.
'''
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import numpy as np
import math

def filler_ratio(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  tokenizer = RegexpTokenizer('uh|ugh|um|like|you know')
  qtokens = tokenizer.tokenize(s.lower())

  if len(tokens) == 0:
    return float(0)
  else:
    return float(len(qtokens)) / float(len(tokens))

def type_token_ratio(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  uniques = []

  for token, count in FreqDist(tokens).items():
    if count == 1:
      uniques.append(token)

  if len(tokens) == 0:
    return float(0)
  else:
    return float(len(uniques)) / float(len(tokens))

def entropy(tokens):
  freqdist = FreqDist(tokens)
  probs = [freqdist.freq(l) for l in freqdist]

  return -sum(p * math.log(p, 2) for p in probs)

def standardized_word_entropy(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)

  if len(tokens) == 0:
    return float(0)
  else:
    if math.log(len(tokens)) == 0:
      return float(0)
    else:
      return entropy(tokens) / math.log(len(tokens))

def question_ratio(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  tokenizer = RegexpTokenizer('Who|What|When|Where|Why|How|\?')
  qtokens = tokenizer.tokenize(s)

  if len(tokens) == 0:
    return float(0)
  else:
    return float(len(qtokens)) / float(len(tokens))

def number_ratio(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)

  tokenizer = RegexpTokenizer('zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|dozen|couple|several|few|\d')
  qtokens = tokenizer.tokenize(s.lower())

  if len(tokens) == 0:
    return float(0)
  else:
    return float(len(qtokens)) / float(len(tokens))

def brunets_index(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  N = float(len(tokens))
  V = float(len(set(tokens)))
  
  if N == 0 or V == 0:
    return float(0)
  else:
    return math.pow(N, math.pow(V, -0.165))

def honores_statistic(s, tokens = None):
  if tokens == None:
    tokens = word_tokenize(s)
    
  uniques = []

  for token, count in FreqDist(tokens).items():
    if count == 1:
      uniques.append(token)

  N  = float(len(tokens))
  V  = float(len(set(tokens)))
  V1 = float(len(uniques))
    
  if N == 0 or V == 0 or V1 == 0:
    return float(0)
  elif V == V1:
    return (100 * math.log(N))
  else:
    return (100 * math.log(N)) / (1 - (V1 / V))

def datewords_freq(importtext):

    text=word_tokenize(importtext.lower())
    datewords=['time','monday','tuesday','wednesday','thursday','friday','saturday','sunday','january','february','march','april','may','june','july','august','september','november','december','year','day','hour','today','month',"o'clock","pm","am"]
    datewords2=list()
    for i in range(len(datewords)):
        datewords2.append(datewords[i]+'s')
    
    datewords=datewords+datewords2                  
    datecount=0

    for i in range(len(text)):
        if text[i].lower() in datewords:
            datecount=datecount+1
            
    datewords=datecount

    try:
        datewordfreq=datecount/len(text)
    except:
        datewordfreq=0

    return datewordfreq

def word_stats(importtext):
    
    text=word_tokenize(importtext)
    #average word length 
    awords=list()
    for i in range(len(text)):
        awords.append(len(text[i]))
    awordlength=np.mean(awords)
    #all words greater than 5 in length
    fivewords= [w for w in text if len(w) > 5]
    fivewordnum=len(fivewords)
    #maximum word length
    vmax=np.amax(awords)
    #minimum word length
    vmin=np.amin(awords)
    #variance of the vocabulary
    vvar=np.var(awords)
    #stdev of vocabulary
    vstd=np.std(awords)
    features=[float(awordlength),float(fivewordnum), float(vmax),float(vmin),float(vvar),float(vstd)]
    return features

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

    #calculate the number of sentences and repeat word avg per sentence 
    sentencenum=len(tenwords)
    repeatavg=repeatnum/sentencenum

    #repeated word freqdist 
    return [repeatavg]

def text_featurize(transcript):
  labels=list()
  features=list()
  # extract features
  features1=[filler_ratio(transcript),
          type_token_ratio(transcript),
          standardized_word_entropy(transcript),
          question_ratio(transcript),
          number_ratio(transcript),
          brunets_index(transcript),
          honores_statistic(transcript),
          datewords_freq(transcript)]
  features2=word_stats(transcript)
  features3=num_sentences(transcript)
  features4=word_repeats(transcript)

  # extract labels 
  labels1=['filler ratio', 'type token ratio', 'standardized word entropy', 'question ratio', 'number ratio', 'Brunets Index',
          'Honores statistic', 'datewords freq']
  labels2=['word number', 'five word count', 'max word length', 'min word length', 'variance of vocabulary', 'std of vocabulary']
  labels3=['sentencenum', 'periods', 'questions', 'interjections']
  labels4=['repeatavg']
  features=features1+features2+features3+features4
  labels=labels1+labels2+labels3+labels4

  return features, labels