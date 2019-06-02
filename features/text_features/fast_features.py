'''
This is the Facebook FastText model:

https://fasttext.cc/docs/en/english-vectors.html
'''
import numpy as np

def fast_featurize(sentence,model):

    size=300
    sentences2=sentence.split()

    w2v_embed=list()
    for i in range(len(sentences2)):
        try:
            print(len(model[sentences2[i]]))
            #print(sentences2[i])
            w2v_embed.append(model[sentences2[i]])
            #print(model[sentences2[i]])
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
        labels.append('fast_feature_%s'%(str(i+1)))

    return out_embed

