'''
train simple classifier.
'''
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from textblob import TextBlob
from operator import itemgetter
import os, json, pickle, datetime, time, shutil
import numpy as np 

# INITIAL FUNCTIONS
#############################################################
def train_sc(alldata,labels,mtype,jsonfile,problemtype,default_features, classes, min_num):

    selectedfeature=default_features + ' (%s)'%(problemtype)
    modelname=jsonfile[0:-5]
    testing_set=0.250
    X_train, X_test, y_train, y_test = train_test_split(alldata, labels, train_size=0.750, test_size=testing_set)
    print(X_train.shape)
    print(X_test.shape)
    training_data='train labels'+'\n\n'+str(y_train)+'\n\n'+'test labels'+'\n\n'+str(y_test)+'\n\n'
    filename=modelname
    start=time.time()
    
    c1=0
    c5=0

    try:
        #decision tree
        classifier2 = DecisionTreeClassifier(random_state=0)
        classifier2.fit(X_train,y_train)
        scores = cross_val_score(classifier2, X_test, y_test,cv=5)
        print('Decision tree accuracy (+/-) %s'%(str(scores.std())))
        c2=scores.mean()
        c2s=scores.std()
        print(c2)
    except:
        c2=0
        c2s=0

    try:
        classifier3 = GaussianNB()
        classifier3.fit(X_train,y_train)
        scores = cross_val_score(classifier3, X_test, y_test,cv=5)
        print('Gaussian NB accuracy (+/-) %s'%(str(scores.std())))
        c3=scores.mean()
        c3s=scores.std()
        print(c3)
    except:
        c3=0
        c3s=0

    try:
        #svc 
        classifier4 = SVC()
        classifier4.fit(X_train,y_train)
        scores=cross_val_score(classifier4, X_test, y_test,cv=5)
        print('SKlearn classifier accuracy (+/-) %s'%(str(scores.std())))
        c4=scores.mean()
        c4s=scores.std()
        print(c4)
    except:
        c4=0
        c4s=0

    try:
        #adaboost
        classifier6 = AdaBoostClassifier(n_estimators=100)
        classifier6.fit(X_train,y_train)
        scores = cross_val_score(classifier6, X_test, y_test,cv=5)
        print('Adaboost classifier accuracy (+/-) %s'%(str(scores.std())))
        c6=scores.mean()
        c6s=scores.std()
        print(c6)
    except:
        c6=0
        c6s=0

    try:
        #gradient boosting 
        classifier7=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        classifier7.fit(X_train,y_train)
        scores = cross_val_score(classifier7, X_test, y_test,cv=5)
        print('Gradient boosting accuracy (+/-) %s'%(str(scores.std())))
        c7=scores.mean()
        c7s=scores.std()
        print(c7)
    except:
        c7=0
        c7s=0

    try:
        #logistic regression
        classifier8=LogisticRegression(random_state=1)
        classifier8.fit(X_train,y_train)
        scores = cross_val_score(classifier8, X_test, y_test,cv=5)
        print('Logistic regression accuracy (+/-) %s'%(str(scores.std())))
        c8=scores.mean()
        c8s=scores.std()
        print(c8)
    except:
        c8=0
        c8s=0

    try:
        #voting 
        classifier9=VotingClassifier(estimators=[('gradboost', classifier7), ('logit', classifier8), ('adaboost', classifier6)], voting='hard')
        classifier9.fit(X_train,y_train)
        scores = cross_val_score(classifier9, X_test, y_test,cv=5)
        print('Hard voting accuracy (+/-) %s'%(str(scores.std())))
        c9=scores.mean()
        c9s=scores.std()
        print(c9)
    except:
        c9=0
        c9s=0

    try:
        #knn
        classifier10=KNeighborsClassifier(n_neighbors=7)
        classifier10.fit(X_train,y_train)
        scores = cross_val_score(classifier10, X_test, y_test,cv=5)
        print('K Nearest Neighbors accuracy (+/-) %s'%(str(scores.std())))
        c10=scores.mean()
        c10s=scores.std()
        print(c10)
    except:
        c10=0
        c10s=0

    try:
        #randomforest
        classifier11=RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        classifier11.fit(X_train,y_train)
        scores = cross_val_score(classifier11, X_test, y_test,cv=5)
        print('Random forest accuracy (+/-) %s'%(str(scores.std())))
        c11=scores.mean()
        c11s=scores.std()
        print(c11)
    except:
        c11=0
        c11s=0

    try:
##        #svm
        classifier12 = svm.SVC(kernel='linear', C = 1.0)
        classifier12.fit(X_train,y_train)
        scores = cross_val_score(classifier12, X_test, y_test,cv=5)
        print('svm accuracy (+/-) %s'%(str(scores.std())))
        c12=scores.mean()
        c12s=scores.std()
        print(c12)
    except:
        c12=0
        c12s=0

    #IF IMBALANCED, USE http://scikit-learn.org/dev/modules/generated/sklearn.naive_bayes.ComplementNB.html

    maxacc=max([c2,c3,c4,c6,c7,c8,c9,c10,c11,c12])
    X_train=list(X_train)
    X_test=list(X_test)
    y_train=list(y_train)
    y_test=list(y_test)
    # if maxacc==c1:
    #     print('most accurate classifier is Naive Bayes'+'with %s'%(selectedfeature))
    #     classifiername='naive-bayes'
    #     classifier=classifier1
    #     #show most important features
    #     classifier1.show_most_informative_features(5)
    if maxacc==c2:
        print('most accurate classifier is Decision Tree'+'with %s'%(selectedfeature))
        classifiername='decision-tree'
        classifier2 = DecisionTreeClassifier(random_state=0)
        classifier2.fit(X_train+X_test,y_train+y_test)
        classifier=classifier2
    elif maxacc==c3:
        print('most accurate classifier is Gaussian NB'+'with %s'%(selectedfeature))
        classifiername='gaussian-nb'
        classifier3 = GaussianNB()
        classifier3.fit(X_train+X_test,y_train+y_test)
        classifier=classifier3
    elif maxacc==c4:
        print('most accurate classifier is SK Learn'+'with %s'%(selectedfeature))
        classifiername='sk'
        classifier4 = SVC()
        classifier4.fit(X_train+X_test,y_train+y_test)
        classifier=classifier4
    elif maxacc==c5:
        print('most accurate classifier is Maximum Entropy Classifier'+'with %s'%(selectedfeature))
        classifiername='max-entropy'
        classifier=classifier5
    #can stop here (c6-c10)
    elif maxacc==c6:
        print('most accuracate classifier is Adaboost classifier'+'with %s'%(selectedfeature))
        classifiername='adaboost'
        classifier6 = AdaBoostClassifier(n_estimators=100)
        classifier6.fit(X_train+X_test,y_train+y_test)
        classifier=classifier6
    elif maxacc==c7:
        print('most accurate classifier is Gradient Boosting '+'with %s'%(selectedfeature))
        classifiername='graidentboost'
        classifier7=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        classifier7.fit(X_train+X_test,y_train+y_test)
        classifier=classifier7
    elif maxacc==c8:
        print('most accurate classifier is Logistic Regression '+'with %s'%(selectedfeature))
        classifiername='logistic_regression'
        classifier8=LogisticRegression(random_state=1)
        classifier8.fit(X_train+X_test,y_train+y_test)
        classifier=classifier8
    elif maxacc==c9:
        print('most accurate classifier is Hard Voting '+'with %s'%(selectedfeature))
        classifiername='hardvoting'
        classifier7=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        classifier7.fit(X_train+X_test,y_train+y_test)
        classifier8=LogisticRegression(random_state=1)
        classifier8.fit(X_train+X_test,y_train+y_test)
        classifier6 = AdaBoostClassifier(n_estimators=100)
        classifier6.fit(X_train+X_test,y_train+y_test)
        classifier9=VotingClassifier(estimators=[('gradboost', classifier7), ('logit', classifier8), ('adaboost', classifier6)], voting='hard')
        classifier9.fit(X_train+X_test,y_train+y_test)
        classifier=classifier9
    elif maxacc==c10:
        print('most accurate classifier is K nearest neighbors '+'with %s'%(selectedfeature))
        classifiername='knn'
        classifier10=KNeighborsClassifier(n_neighbors=7)
        classifier10.fit(X_train+X_test,y_train+y_test)
        classifier=classifier10
    elif maxacc==c11:
        print('most accurate classifier is Random forest '+'with %s'%(selectedfeature))
        classifiername='randomforest'
        classifier11=RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        classifier11.fit(X_train+X_test,y_train+y_test)
        classifier=classifier11
    elif maxacc==c12:
        print('most accurate classifier is SVM '+' with %s'%(selectedfeature))
        classifiername='svm'
        classifier12 = svm.SVC(kernel='linear', C = 1.0)
        classifier12.fit(X_train+X_test,y_train+y_test)
        classifier=classifier12

    modeltypes=['decision-tree','gaussian-nb','sk','adaboost','gradient boosting','logistic regression','hard voting','knn','random forest','svm']
    accuracym=[c2,c3,c4,c6,c7,c8,c9,c10,c11,c12]
    accuracys=[c2s,c3s,c4s,c6s,c7s,c8s,c9s,c10s,c11s,c12s]
    model_accuracy=list()
    for i in range(len(modeltypes)):
        model_accuracy.append([modeltypes[i],accuracym[i],accuracys[i]])

    model_accuracy.sort(key=itemgetter(1))
    endlen=len(model_accuracy)

    print('saving classifier to disk')
    f=open(modelname+'.pickle','wb')
    pickle.dump(classifier,f)
    f.close()

    end=time.time()

    execution=end-start
    
    print('summarizing session...')

    accstring=''
    
    for i in range(len(model_accuracy)):
        accstring=accstring+'%s: %s (+/- %s)\n'%(str(model_accuracy[i][0]),str(model_accuracy[i][1]),str(model_accuracy[i][2]))

    training=len(X_train)
    testing=len(y_train)
    
    summary='SUMMARY OF MODEL SELECTION \n\n'+'WINNING MODEL: \n\n'+'%s: %s (+/- %s) \n\n'%(str(model_accuracy[len(model_accuracy)-1][0]),str(model_accuracy[len(model_accuracy)-1][1]),str(model_accuracy[len(model_accuracy)-1][2]))+'MODEL FILE NAME: \n\n %s.pickle'%(filename)+'\n\n'+'DATE CREATED: \n\n %s'%(datetime.datetime.now())+'\n\n'+'EXECUTION TIME: \n\n %s\n\n'%(str(execution))+'GROUPS: \n\n'+str(classes)+'\n'+'('+str(min_num)+' in each class, '+str(int(testing_set*100))+'% used for testing)'+'\n\n'+'TRAINING SUMMARY:'+'\n\n'+training_data+'FEATURES: \n\n %s'%(selectedfeature)+'\n\n'+'MODELS, ACCURACIES, AND STANDARD DEVIATIONS: \n\n'+accstring+'\n\n'+'(C) 2019, NeuroLex Laboratories'

    data={'sample type': problemtype,
        'feature_set':default_features,
        'model name':modelname+'.pickle',
        'accuracy':model_accuracy[len(model_accuracy)-1][1],
        'deviation': model_accuracy[len(model_accuracy)-1][2],
        'model type':'sc_'+classifiername,
        }

    # write to .JSON and move to proper directory...
    g=open(modelname+'.txt','w')
    g.write(summary)
    g.close()

    g2=open(modelname+'.json','w')
    json.dump(data,g2)
    g2.close()

    
    cur_dir2=os.getcwd()
    try:
        os.chdir(problemtype+'_models')
    except:
        os.mkdir(problemtype+'_models')
        os.chdir(problemtype+'_models')

    # now move all the files over to proper model directory 
    shutil.move(cur_dir2+'/'+modelname+'.json', os.getcwd()+'/'+modelname+'.json')
    shutil.move(cur_dir2+'/'+modelname+'.pickle', os.getcwd()+'/'+modelname+'.pickle')
    shutil.move(cur_dir2+'/'+modelname+'.txt', os.getcwd()+'/'+modelname+'.txt')

