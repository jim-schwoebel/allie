'''
Train_audioTPOT.py

Takes in N number of classes and builds a regression model
or classification model from custom inputs.

Note this strategy can be used to optimize machine learning model
pipelines.

I have here the basic modeling script, assuming the classes already have 
been specified in a .JSON document.
'''

import json, os, random
import numpy as np
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

## initialize directories and classes
model_dir=os.getcwd()+'/models/'
data_dir=os.getcwd()+'/data/'

os.chdir(data_dir)
mtype=input('classification (c) or regression (r) problem? \n').lower().replace(' ','')
while mtype not in ['c','r', 'classification','regression']:
    print('input not recognized')
    mtype=input('is this classification (c) or regression (r) problem? \n').lower().replace(' ','')

numclasses=input('how many classes are you training?')
classes=list()

for i in range(int(numclasses)):
    classes.append(input('what is the name of class %s?'%(str(i+1))))

jsonfile=''
for i in range(len(classes)):
    if i==0:
        jsonfile=classes[i]
    else:
        jsonfile=jsonfile+'_'+classes[i]

jsonfile=jsonfile+'.json'

try:
    g=json.load(open(jsonfile))
    alldata=list()
    labels=list()
    lengths=list()

    # check to see all classes are same length and reshape if necessary
    for i in range(len(classes)):
        class_=g[classes[i]]
        lengths.append(len(class_))

    lengths=np.array(lengths)
    minlength=np.amin(lengths)

    # now load all the classes
    for i in range(len(classes)):
        class_=g[classes[i]]
        random.shuffle(class_)

        if len(class_) > minlength:
            print('%s greater than class, equalizing...')
            class_=class_[0:minlength]

        for j in range(len(class_)):
            alldata.append(class_[i])
            labels.append(i)

    os.chdir(model_dir)

    alldata=np.asarray(alldata)
    labels=np.asarray(labels)

    # get train and test data 
    X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)
    if mtype in [' classification', 'c']:
        tpot=TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs=-1)
        tpotname='%s_tpotclassifier.py'%(jsonfile[0:-5])
    elif mtype in ['regression','r']:
        tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
        tpotname='%s_tpotregression.py'%(jsonfile[0:-5])
    tpot.fit(X_train, y_train)
    accuracy=tpot.score(X_test,y_test)
    tpot.export(tpotname)

    # export data to .json format 
    data={
        'data': alldata.tolist(),
        'labels': labels.tolist(),
    }

    jsonfilename='%s_.json'%(tpotname[0:-3])
    jsonfile=open(jsonfilename,'w')
    json.dump(data,jsonfile)
    jsonfile.close()

    # now edit the file and run it 
    g=open(tpotname).read()
    g=g.replace("import numpy as np", "import numpy as np \nimport json, pickle")
    g=g.replace("tpot_data = pd.read_csv(\'PATH/TO/DATA/FILE\', sep=\'COLUMN_SEPARATOR\', dtype=np.float64)","g=json.load(open('%s'))\ntpot_data=g['labels']"%(jsonfilename))
    g=g.replace("features = tpot_data.drop('target', axis=1).values","features=g['data']\n")
    g=g.replace("tpot_data['target'].values", "tpot_data")
    g=g.replace("results = exported_pipeline.predict(testing_features)", "print('saving classifier to disk')\nf=open('%s','wb')\npickle.dump(exported_pipeline,f)\nf.close()"%(jsonfilename[0:-6]+'.pickle'))
    g1=g.find('exported_pipeline = ')
    g2=g.find('exported_pipeline.fit(training_features, training_target)')
    modeltype=g[g1:g2]
    os.remove(tpotname)
    t=open(tpotname,'w')
    t.write(g)
    t.close()
    os.system('python3 %s'%(tpotname))

    # now write an accuracy label 
    os.remove(jsonfilename)

    jsonfilename='%s.json'%(tpotname[0:-3])
    print('saving .JSON file (%s)'%(jsonfilename))
    jsonfile=open(jsonfilename,'w')
    if mtype in ['classification', 'c']:
        data={
            'model name':jsonfilename[0:-5]+'.pickle',
            'accuracy':accuracy,
            'model type':'TPOTclassification_'+modeltype,
        }
    elif mtype in ['regression', 'r']:
        data={
            'model name':jsonfilename[0:-5]+'.pickle',
            'accuracy':accuracy,
            'model type':'TPOTregression_'+modeltype,
        }

    json.dump(data,jsonfile)
    jsonfile.close()
                        
except:    
    print('error, please put %s in %s'%(jsonfile, data_dir))
    print('note this can be done with train_audioclassify.py script')