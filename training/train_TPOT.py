import os, sys, pickle, json, random, shutil, time
import numpy as np
import matplotlib.pyplot as plt
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def train_TPOT(alldata, labels, mtype, jsonfile, problemtype, default_features, settings):
    # get train and test data 
    X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=0.250)
    modelname=jsonfile[0:-5]+'_'+str(default_features).replace("'",'').replace('"','')
    if mtype in [' classification', 'c']:
        tpot=TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs=-1)
        tpotname='%s_tpotclassifier.py'%(modelname)
    elif mtype in ['regression','r']:
        tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
        tpotname='%s_tpotregression.py'%(modelname)
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
    g=g.replace("tpot_data = pd.read_csv(\'PATH/TO/DATA/FILE\', sep=\'COLUMN_SEPARATOR\', dtype=np.float64)","g=json.load(open('%s'))\ntpot_data=np.array(g['labels'])"%(jsonfilename))
    g=g.replace("features = tpot_data.drop('target', axis=1)","features=np.array(g['data'])\n")
    g=g.replace("tpot_data['target'].values", "tpot_data")
    g=g.replace("results = exported_pipeline.predict(testing_features)", "print('saving classifier to disk')\nf=open('%s','wb')\npickle.dump(exported_pipeline,f)\nf.close()"%(jsonfilename[0:-6]+'.pickle'))
    g1=g.find('exported_pipeline = ')
    g2=g.find('exported_pipeline.fit(training_features, training_target)')
    g=g.replace('.values','')
    modeltype=g[g1:g2]
    os.remove(tpotname)
    t=open(tpotname,'w')
    t.write(g)
    t.close()
    print('')
    os.system('python3 %s'%(tpotname))

    # now write an accuracy label 
    os.remove(jsonfilename)

    jsonfilename='%s.json'%(tpotname[0:-3])
    print('saving .JSON file (%s)'%(jsonfilename))
    jsonfile=open(jsonfilename,'w')
    if mtype in ['classification', 'c']:
        data={'sample type': problemtype,
            'feature_set':default_features,
            'model name':jsonfilename[0:-5]+'.pickle',
            'accuracy':accuracy,
            'model type':'TPOTclassification_'+modeltype,
            'settings': settings,
        }
    elif mtype in ['regression', 'r']:
        data={'sample type': problemtype,
            'feature_set':default_features,
            'model name':jsonfilename[0:-5]+'.pickle',
            'accuracy':accuracy,
            'model type':'TPOTregression_'+modeltype,
            'settings': settings,
        }

    json.dump(data,jsonfile)
    jsonfile.close()

    cur_dir2=os.getcwd()
    try:
    	os.chdir(problemtype+'_models')
    except:
    	os.mkdir(problemtype+'_models')
    	os.chdir(problemtype+'_models')

    # now move all the files over to proper model directory 
    shutil.move(cur_dir2+'/'+jsonfilename, os.getcwd()+'/'+jsonfilename)
    shutil.move(cur_dir2+'/'+tpotname, os.getcwd()+'/'+tpotname)
    shutil.move(cur_dir2+'/'+jsonfilename[0:-5]+'.pickle', os.getcwd()+'/'+jsonfilename[0:-5]+'.pickle')

    # get model_name 
    model_name=jsonfilename[0:-5]+'.pickle'
    model_dir=os.getcwd()
    
    return model_name, model_dir