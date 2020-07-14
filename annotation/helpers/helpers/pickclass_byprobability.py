import pickle, os, json
import numpy as np

def pick_class(classlist):

    names=['teens','twenties','thirties','fourties','fifties','sixties','seventies']
    probabilities=[.0666,.48888,.2296296,.08888,.08888,.0296,.0074]
    freqs=list()

    for i in range(len(classlist)):
        try:
            index=names.index(classlist[i])
            freq=probabilities[index]
            freqs.append(freqs)
        except:
            freq=0
            freqs.append(freq)

    #now pick the maxinum frequency
    maxfreq=np.amax(freqs)
    index=freqs.index(maxfreq)

    return classlist[index]

def classify(features):
    
    listdir=os.listdir()

    model_list=['teens.pickle','twenties.pickle','thirties.pickle','fourties.pickle',
                'fifties.pickle','sixties.pickle','seventies.pickle']

    classlist=list()
    model_acc=list()
    deviations=list()
    modeltypes=list()
    modelslist=list()

    for i in range(len(model_list)):

        modelname=model_list[i]
        loadmodel=open(modelname,'rb')
        model=pickle.load(loadmodel)
        loadmodel.close()
        output=str(model.predict(features)[0])
        classname=output
        if classname.count('controls')>0:
            pass
        else:
            classlist.append(classname)

    if len(classlist)>1:
        winclass=pick_class(classlist)
        modelslist.append(winclass+'.pickle')
        g=json.load(open(winclass+'.json'))
        model_acc.append(g['accuracy'])
        deviations.append(g['deviation'])
        modeltypes.append(g['modeltype'])
    elif len(classlist)==1:
        winclass=classlist[0]
        modelslist.append(winclass+'.pickle')
        g=json.load(open(winclass+'.json'))
        model_acc.append(g['accuracy'])
        deviations.append(g['deviation'])
        modeltypes.append(g['modeltype'])
    else:
        winclass='n/a'
        g=json.load(open(winclass+'.json'))
        model_acc.append(0)
        deviations.append(0)
        modeltypes.append('n/a')
    
    return winclass

