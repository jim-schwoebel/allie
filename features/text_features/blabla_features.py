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
                   
		   
Featurize folders of text files if default_text_features = ['glove_features']

This uses a GloVE embedding with 100 dimensions:
https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
'''
import os, uuid
import numpy as np
import pandas as pd 

def setup_blabla():
  # assumes you have blabla installed 
  os.system('pip3 install blabla')
  # install corenlp
  os.chdir('helpers')
  os.chdir('blabla')
  os.system('./setup_corenlp.sh')
  os.system('export CORENLP_HOME=/Users/jim/corenlp')
  '''
  CoreNLP (english) successfully installed at /Users/jim/corenlp
  Now and in the future, run 'export CORENLP_HOME=/Users/jim/corenlp' before using BlaBla or add this command to your .bashrc/.profile or equivalent file
  '''
  # make sure javac is installed as well (assumes you have done this)
  os.system('javac -version')

def blabla_featurize(transcript):

    curdir=os.getcwd()
    text_folder=str(uuid.uuid4())
    os.mkdir(text_folder)
    text_folderpath=os.getcwd()+'/'+text_folder
    os.chdir(text_folder)
    g=open('transcript.txt','w')
    g.write(transcript)
    g.close()

    g=open('transcript2.txt','w')
    g.write(transcript)
    g.close() 

    os.chdir(curdir)
    os.system('blabla compute-features -F helpers/blabla/example_configs/features.yaml -S helpers/blabla/stanza_config/stanza_config.yaml -i %s -o %s/blabla_features.csv -format string'%(text_folderpath, text_folderpath))
    
    os.chdir(text_folder)
    g=pd.read_csv('blabla_features.csv')
    features=list(g.iloc[0,:][0:-1])
    labels=list(g)[0:-1]
    os.chdir(curdir)

    return features, labels 

features, labels = blabla_featurize('This is the coolest transcript ever. This is the coolest transcript ever.')
print(features)
print(labels)
