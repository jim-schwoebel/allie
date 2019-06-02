'''
This is the Google W2V embedding (train_textclassify_3.py)

Following this tutorial:
https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
'''
import numpy as np

def w2v_featurize(transcript,model):

    sentences2=sentence.split()
    size=300

    w2v_embed=list()
    for i in range(len(sentences2)):
        try:
            print(len(model[sentences2[i]]))
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
        labels.append('w2v_feature_%s'%(str(i+1)))

    return out_embed


