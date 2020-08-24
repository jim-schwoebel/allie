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


|  ___|       | |                        / _ \ | ___ \_   _|  _ 
| |_ ___  __ _| |_ _   _ _ __ ___  ___  / /_\ \| |_/ / | |   (_)
|  _/ _ \/ _` | __| | | | '__/ _ \/ __| |  _  ||  __/  | |      
| ||  __/ (_| | |_| |_| | | |  __/\__ \ | | | || |    _| |_   _ 
\_| \___|\__,_|\__|\__,_|_|  \___||___/ \_| |_/\_|    \___/  (_)
                                                                
                                                                
  ___            _ _       
 / _ \          | (_)      
/ /_\ \_   _  __| |_  ___  
|  _  | | | |/ _` | |/ _ \ 
| | | | |_| | (_| | | (_) |
\_| |_/\__,_|\__,_|_|\___/ 
                           

This will featurize folders of audio files if the default_audio_features = ['audiotext_features']

Featurizes data with text feautures extracted from the transcript.
These text features include nltk_features, textacy_features, spacy_features, and text_features.
'''

import librosa_features as lf 
import helpers.transcribe as ts
import numpy as np
import random, math, os, sys, json, time

def prev_dir(directory):
    g=directory.split('/')
    dir_=''
    for i in range(len(g)):
        if i != len(g)-1:
            if i==0:
                dir_=dir_+g[i]
            else:
                dir_=dir_+'/'+g[i]
    # print(dir_)
    return dir_

directory=os.getcwd()
prevdir=prev_dir(directory)
sys.path.append(prevdir+'/text_features')
import nltk_features as nf 
import textacy_features as tfe
import spacy_features as spf
import text_features as tfea

def audiotext_featurize(wavfile, transcript):

    try:
        # get features 
        try:
            nltk_features, nltk_labels=nf.nltk_featurize(transcript)
        except:
            nltk_labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'space', 'numbers', 'capletters', 'cc', 'cd', 'dt', 'ex', 'in', 'jj', 'jjr', 'jjs', 'ls', 'md', 'nn', 'nnp', 'nns', 'pdt', 'pos', 'prp', 'prp2', 'rbr', 'rbs', 'rp', 'to', 'uh', 'vb', 'vbd', 'vbg', 'vbn', 'vbp', 'vbz', 'wdt', 'wp', 'wrb', 'polarity', 'subjectivity', 'repeat']
            nltk_features=list(np.zeros(len(nltk_labels)))

        try:
            textacy_features, textacy_labels=tfe.textacy_featurize(transcript)
        except:
            textacy_labels=['uniquewords', 'n_sents', 'n_words', 'n_chars', 'n_syllables', 'n_unique_words', 'n_long_words', 'n_monosyllable_words', 'n_polysyllable_words', 'flesch_kincaid_grade_level', 'flesch_kincaid_grade_level', 'flesch_reading_ease', 'smog_index', 'gunning_fog_index', 'coleman_liau_index', 'automated_readability_index', 'lix', 'gulpease_index', 'wiener_sachtextformel']
            textacy_features=list(np.zeros(len(textacy_labels)))

        try:
            spacy_features, spacy_labels=spf.spacy_featurize(transcript)
        except:
            spacy_labels=['PROPN', 'ADP', 'DET', 'NUM', 'PUNCT', 'SPACE', 'VERB', 'NOUN', 'ADV', 'CCONJ', 'PRON', 'ADJ', 'SYM', 'PART', 'INTJ', 'X', 'pos_other', 'NNP', 'IN', 'DT', 'CD', 'NNPS', ',', '_SP', 'VBZ', 'NN', 'RB', 'CC', '', 'NNS', '.', 'PRP', 'MD', 'VB', 'HYPH', 'VBD', 'JJ', ':', '-LRB-', '$', '-RRB-', 'VBG', 'VBN', 'NFP', 'RBR', 'POS', 'VBP', 'RP', 'JJS', 'PRP$', 'EX', 'JJR', 'WP', 'WDT', 'TO', 'WRB', "''", '``', 'PDT', 'AFX', 'RBS', 'UH', 'WP$', 'FW', 'XX', 'SYM', 'LS', 'ADD', 'tag_other', 'compound', 'ROOT', 'prep', 'det', 'pobj', 'nummod', 'punct', '', 'nsubj', 'advmod', 'cc', 'conj', 'aux', 'dobj', 'nmod', 'acl', 'appos', 'npadvmod', 'amod', 'agent', 'case', 'intj', 'prt', 'pcomp', 'ccomp', 'attr', 'dep', 'acomp', 'poss', 'auxpass', 'expl', 'mark', 'nsubjpass', 'quantmod', 'advcl', 'relcl', 'oprd', 'neg', 'xcomp', 'csubj', 'predet', 'parataxis', 'dative', 'preconj', 'csubjpass', 'meta', 'dep_other', '\ufeffXxx', 'Xxxxx', 'XXxxx', 'xx', 'X', 'Xxxx', 'Xxx', ',', '\n\n', 'xXxxx', 'xxx', 'xxxx', '\n', '.', ' ', '-', 'xxx.xxxx.xxx', '\n\n\n', ':', '\n    ', 'dddd', '[', '#', 'dd', ']', 'd', 'XXX-d', '*', 'XXXX', 'XX', 'XXX', '\n\n\n\n', 'Xx', '\n\n\n    ', '--', '\n\n    ', '    ', '   ', '  ', "'x", 'x', 'X.', 'xxx--', ';', 'Xxx.', '(', ')', "'", '“', '”', 'Xx.', '!', "'xx", 'xx!--Xxx', "x'xxxx", '?', '_', "x'x", "x'xx", "Xxx'xxxx", 'Xxxxx--', 'xxxx--', '--xxxx', 'X--', 'xx--', 'xxxx”--xxx', 'xxx--“xxxx', "Xxx'x", ';--', 'xxx--_xxx', "xxx'x", 'xxx!--xxxx', 'xxxx?--_Xxx', "Xxxxx'x", 'xxxx--“xxxx', "xxxx'xxx", '--Xxxxx', ',--', '?--', 'xx--“xx', 'xx!--X', '.--', 'xxx--“xxx', ':--', 'Xxxxx--“xxxx', 'xxxx!--xxxx', 'xx”--xxx', 'xxxx--_xxx', 'xxxx--“xxx', '--xx', '--X', 'xxxx!--Xxx', '--xxx', 'xxx_.', 'xxxx--_xx', 'xxxx--_xx_xxxx', 'xx!--xxxx', 'xxxx!--xx', "X'xx", "xxxx'x", "X_'x", "xxx'xxx", '--Xxxx', "X'Xxxxx", "Xx'xxxx", '--Xxx', 'xxxx”--xxxx', 'xxxx!--', 'xxxx--“x', 'Xxxx!--Xxxx', 'xxx!--Xxx', 'Xxxxx.', 'xxxx_.', 'xx--“Xxxx', '\n\n   ', 'Xxxxx”--xxx', 'xxxx”--xx', 'xxxx--“xx', "Xxxxx!--Xxx'x", "X'xxxx", 'Xxxxx?--', '--Xx', 'xxxx!”--Xx', "xxxx--“X'x", "xxxx'", 'xxx.--“Xxxx', 'xxxx--“X', 'xxxx!--X', 'Xxx”--xx', 'xxx”--xxx', 'xxx-_xxx', "x'Xxxxx", 'Xxxxx!--X', 'Xxxxx!--Xxx', 'dd-d.xxx', 'xxxx://xxx.xxxx.xxx/d/dd/', 'xXxxxx', 'xxxx://xxxx.xxx/xxxx', 'd.X.', '/', 'd.X.d', 'd.X', '%', 'Xd', 'xxxx://xxx.xxxx.xxx', 'ddd(x)(d', 'X.X.', 'ddd', 'xxxx@xxxx.xxx', 'xxxx://xxxx.xxx', '$', 'd,ddd', 'shape_other', 'mean sentence polarity', 'std sentence polarity', 'max sentence polarity', 'min sentence polarity', 'median sentence polarity', 'mean sentence subjectivity', 'std sentence subjectivity', 'max sentence subjectivity', 'min sentence subjectivity', 'median sentence subjectivity', 'character count', 'word count', 'sentence number', 'words per sentence', 'unique chunk noun text', 'unique chunk root text', 'unique chunk root head text', 'chunkdep ROOT', 'chunkdep pobj', 'chunkdep nsubj', 'chunkdep dobj', 'chunkdep conj', 'chunkdep appos', 'chunkdep attr', 'chunkdep nsubjpass', 'chunkdep dative', 'chunkdep pcomp', 'number of named entities', 'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
            spacy_features=list(np.zeros(len(spacy_labels)))

        try:
            text_features,text_labels=tfea.text_featurize(transcript)
        except:
            text_labels=['filler ratio', 'type token ratio', 'standardized word entropy', 'question ratio', 'number ratio', 'Brunets Index', 'Honores statistic', 'datewords freq', 'word number', 'five word count', 'max word length', 'min word length', 'variance of vocabulary', 'std of vocabulary', 'sentencenum', 'periods', 'questions', 'interjections', 'repeatavg']
            text_features=list(np.zeros(len(text_labels)))
            
        # concatenate feature arrays 
        features=np.append(np.array(nltk_features),np.array(textacy_features))
        features=np.append(features,np.array(spacy_features))
        features=np.append(features, np.array(text_features))
        
        # concatenate labels
        labels=nltk_labels+textacy_labels+spacy_labels+text_labels
    except:
        labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
                'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'space', 'numbers', 'capletters', 
                'cc', 'cd', 'dt', 'ex', 'in', 'jj', 'jjr', 'jjs', 'ls', 'md', 'nn', 'nnp', 'nns', 'pdt', 
                'pos', 'prp', 'prp2', 'rbr', 'rbs', 'rp', 'to', 'uh', 'vb', 'vbd', 'vbg', 'vbn', 
                'vbp', 'vbz', 'wdt', 'wp', 'wrb', 'polarity', 'subjectivity', 'repeat', 
                'uniquewords', 'n_sents', 'n_words', 'n_chars', 'n_syllables', 'n_unique_words', 
                'n_long_words', 'n_monosyllable_words', 'n_polysyllable_words', 'flesch_kincaid_grade_level', 
                'flesch_kincaid_grade_level', 'flesch_reading_ease', 'smog_index', 'gunning_fog_index', 'coleman_liau_index', 
                'automated_readability_index', 'lix', 'gulpease_index', 'wiener_sachtextformel', 'PROPN', 'ADP', 
                'DET', 'NUM', 'PUNCT', 'SPACE', 'VERB', 'NOUN', 'ADV', 'CCONJ', 'PRON', 'ADJ', 'SYM', 'PART', 
                'INTJ', 'X', 'pos_other', 'NNP', 'IN', 'DT', 'CD', 'NNPS', ',', '_SP', 'VBZ', 'NN', 'RB', 'CC', 
                '', 'NNS', '.', 'PRP', 'MD', 'VB', 'HYPH', 'VBD', 'JJ', ':', '-LRB-', '$', '-RRB-', 'VBG', 
                'VBN', 'NFP', 'RBR', 'POS', 'VBP', 'RP', 'JJS', 'PRP$', 'EX', 'JJR', 'WP', 'WDT', 'TO', 'WRB', 
                "''", '``', 'PDT', 'AFX', 'RBS', 'UH', 'WP$', 'FW', 'XX', 'SYM', 'LS', 'ADD', 'tag_other', 
                'compound', 'ROOT', 'prep', 'det', 'pobj', 'nummod', 'punct', '', 'nsubj', 'advmod', 'cc', 
                'conj', 'aux', 'dobj', 'nmod', 'acl', 'appos', 'npadvmod', 'amod', 'agent', 'case', 'intj', 
                'prt', 'pcomp', 'ccomp', 'attr', 'dep', 'acomp', 'poss', 'auxpass', 'expl', 'mark', 'nsubjpass', 
                'quantmod', 'advcl', 'relcl', 'oprd', 'neg', 'xcomp', 'csubj', 'predet', 'parataxis', 'dative', 
                'preconj', 'csubjpass', 'meta', 'dep_other', '\ufeffXxx', 'Xxxxx', 'XXxxx', 'xx', 'X', 'Xxxx', 'Xxx', 
                ',', '\n\n', 'xXxxx', 'xxx', 'xxxx', '\n', '.', ' ', '-', 'xxx.xxxx.xxx', '\n\n\n', ':', '\n    ', 
                'dddd', '[', '#', 'dd', ']', 'd', 'XXX-d', '*', 'XXXX', 'XX', 'XXX', '\n\n\n\n', 'Xx', '\n\n\n    ', 
                '--', '\n\n    ', '    ', '   ', '  ', "'x", 'x', 'X.', 'xxx--', ';', 'Xxx.', '(', ')', "'", '“', '”', 
                'Xx.', '!', "'xx", 'xx!--Xxx', "x'xxxx", '?', '_', "x'x", "x'xx", "Xxx'xxxx", 'Xxxxx--', 'xxxx--', 
                '--xxxx', 'X--', 'xx--', 'xxxx”--xxx', 'xxx--“xxxx', "Xxx'x", ';--', 'xxx--_xxx', "xxx'x", 'xxx!--xxxx',
                'xxxx?--_Xxx', "Xxxxx'x", 'xxxx--“xxxx', "xxxx'xxx", '--Xxxxx', ',--', '?--', 'xx--“xx', 'xx!--X', 
                '.--', 'xxx--“xxx', ':--', 'Xxxxx--“xxxx', 'xxxx!--xxxx', 'xx”--xxx', 'xxxx--_xxx', 'xxxx--“xxx', 
                '--xx', '--X', 'xxxx!--Xxx', '--xxx', 'xxx_.', 'xxxx--_xx', 'xxxx--_xx_xxxx', 'xx!--xxxx', 
                'xxxx!--xx', "X'xx", "xxxx'x", "X_'x", "xxx'xxx", '--Xxxx', "X'Xxxxx", "Xx'xxxx", '--Xxx', 
                'xxxx”--xxxx', 'xxxx!--', 'xxxx--“x', 'Xxxx!--Xxxx', 'xxx!--Xxx', 'Xxxxx.', 'xxxx_.', 
                'xx--“Xxxx', '\n\n   ', 'Xxxxx”--xxx', 'xxxx”--xx', 'xxxx--“xx', "Xxxxx!--Xxx'x", "X'xxxx", 
                'Xxxxx?--', '--Xx', 'xxxx!”--Xx', "xxxx--“X'x", "xxxx'", 'xxx.--“Xxxx', 'xxxx--“X', 'xxxx!--X', 
                'Xxx”--xx', 'xxx”--xxx', 'xxx-_xxx', "x'Xxxxx", 'Xxxxx!--X', 'Xxxxx!--Xxx', 'dd-d.xxx', 
                'xxxx://xxx.xxxx.xxx/d/dd/', 'xXxxxx', 'xxxx://xxxx.xxx/xxxx', 'd.X.', '/', 'd.X.d',
                'd.X', '%', 'Xd', 'xxxx://xxx.xxxx.xxx', 'ddd(x)(d', 'X.X.', 'ddd', 'xxxx@xxxx.xxx', 
                'xxxx://xxxx.xxx', '$', 'd,ddd', 'shape_other', 'mean sentence polarity', 'std sentence polarity', 
                'max sentence polarity', 'min sentence polarity', 'median sentence polarity', 'mean sentence subjectivity', 
                'std sentence subjectivity', 'max sentence subjectivity', 'min sentence subjectivity', 'median sentence subjectivity', 
                'character count', 'word count', 'sentence number', 'words per sentence', 'unique chunk noun text', 
                'unique chunk root text', 'unique chunk root head text', 'chunkdep ROOT', 'chunkdep pobj', 
                'chunkdep nsubj', 'chunkdep dobj', 'chunkdep conj', 'chunkdep appos', 'chunkdep attr', 
                'chunkdep nsubjpass', 'chunkdep dative', 'chunkdep pcomp', 'number of named entities', 
                'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 
                'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'filler ratio', 'type token ratio', 
                'standardized word entropy', 'question ratio', 'number ratio', 'Brunets Index', 'Honores statistic', 
                'datewords freq', 'word number', 'five word count', 'max word length', 'min word length', 'variance of vocabulary', 
                'std of vocabulary', 'sentencenum', 'periods', 'questions', 'interjections', 'repeatavg']

        features=list(np.zeros(len(labels)))

    return features, labels 