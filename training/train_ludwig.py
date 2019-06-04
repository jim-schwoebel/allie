'''
ludwig_text
'''
import os, csv, json, random
import numpy as np
from ludwig.api import LudwigModel

# write to .csv
def write_csv(filename, data):
        # taken from https://realpython.com/python-csv/
        labels=list(data)
        print(labels)
        length=len(data[labels[0]])

        # write to csv_file in key/value pair 
        with open(filename, mode='w') as csv_file:
            fieldnames = labels
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            # get the labels
            for i in range(length):
                element=dict()
                for j in range(len(labels)):
                    element[labels[j]]=data[labels[j]][i]
                    print(data[labels[j]][i])

                print(element)
                writer.writerow(element)

        return filename

def make_yaml(data, input_type, output_type, epochs):

    #  make the yaml file 
    # assume inputs in first column and outputs in second column 
    data=list(data)
    input_name=data[0]
    output_name=data[1]
    print('making yaml file --> model_definition.yaml')
    inputs='input features:\n-\n  name: %s\n  type: %s\n'%(input_name, input_type)
    outputs='output features:\n-\n  name: %s\n  type: %s\n'%(output_name, output_type)
    training='training:\n epochs: %s'%(epochs)
    text=inputs+outputs+training

    g=open('model_definition.yaml','w')
    g.write(text)
    g.close()

## initialize directories and classes
model_dir=os.getcwd()+'/models/'
data_dir=os.getcwd()+'/data/'

os.chdir(data_dir)
mtype=input('classification (c) or regression (r) problem? \n').lower().replace(' ','')
while mtype not in ['c','r', 'classification','regression']:
    print('input not recognized')
    mtype=input('is this classification (c) or regression (r) problem? \n').lower().replace(' ','')

outputname=input('what is the output model name? (e.g. gender if a male/female classification problem)')
featurename=input('what is the feature embedding name (e.g. leave blank for mfcc coefficients)')
if featurename=='':
    featurename='mfcc_coefficients'
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

#try:
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

# export data to .json format 
data={
    'data': alldata.tolist(),
    'labels': labels.tolist(),
}

jsonfilename='%s_.json'%(jsonfile[0:-5]+"_ludwig")
jsonfile=open(jsonfilename,'w')
json.dump(data,jsonfile)
jsonfile.close()

filename='%s.csv'%(jsonfilename[0:-5])
filename=write_csv(filename, data)

# now make a model_definition.yaml
epochs='10'
input_type='numerical'
output_type='category'
feature_inputs=list()

# this would actually go very nicely with the featurization scripts I made for NLTK, Spacy, etc. 
# features, labels --> labels as numerical features (take from the Voicebook) 

for i in range(len(alldata[0])):
    tempdict=dict()
    tempdict['name']=featurename+'_'+str(i)
    tempdict['type']='numerical'
    feature_inputs.append(tempdict)

model_definition = {'input_features': feature_inputs, 'output_features': [{outputname: outputname, 'type': 'category'}]}
ludwig_model = LudwigModel(model_definition)
train_stats = ludwig_model.train(data_csv=os.getcwd()+'/'+filename)

# jsonfilename=jsonfilename
# print('saving .JSON file (%s)'%(jsonfilename))
# jsonfile=open(jsonfilename,'w')
# if mtype in ['classification', 'c']:
#     data={
#         'model name':jsonfilename[0:-5]+'.pickle',
#         'accuracy':accuracy,
#         'model type':'TPOTclassification_'+modeltype,
#     }
# elif mtype in ['regression', 'r']:
#     data={
#         'model name':jsonfilename[0:-5]+'.pickle',
#         'accuracy':accuracy,
#         'model type':'TPOTregression_'+modeltype,
#     }

# json.dump(data,jsonfile)
# jsonfile.close()
                        
# except:    
#     print('error, please put %s in %s'%(jsonfile, data_dir))
#     print('note this can be done with train_audioclassify.py script')


########################


# make_yaml(data, input_type, output_type, epochs)
# # clone the repo in folder 
# if 'ludwig' not in os.listdir():
#     os.system('git clone https://github.com/uber/ludwig')

# # make a .CSV file from the data properly formatted
# command='python3 ludwig/ludwig/train.py ludwig train –data_csv –md "{input_features: [{name: name, type: text}], output_features: [{name: gender, type: category}]}"'
# print(command)
# print(os.getcwd())
# os.system(command)
