'''
This uses a GloVE embedding (train_textclassify_2.py)

Following this tutorial:
https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
'''

import numpy as np

def glove_featurize(transcript,model):

    # set to 100 size 
    sentences2=transcript.split()
    size=100
    w2v_embed=list()
    for i in range(len(sentences2)):
        try:
            w2v_embed.append(model[sentences2[i]])
        except:
            #pass if there is an error to not distort averages... :)
            pass

    out_embed=np.zeros(size)
    for j in range(len(w2v_embed)):
        out_embed=out_embed+np.array(w2v_embed[j])

    out_embed=(1/len(w2v_embed))*out_embed
    features=out_embed
    labels=list()
    for i in range(len(features)):
        labels.append('glove_feature_%s'%(str(i+1)))

    return features, labels 


