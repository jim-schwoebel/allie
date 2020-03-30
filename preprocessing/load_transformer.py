'''
[58.0, 13.0, 36.0, 41.0, 128.0, 15.0, 16.0, 40.0, 162.0, 1.0, 14.0, 
30.0, 29.0, 69.0, 80.0, 14.0, 1.0, 58.0, 65.0, 80.0, 60.0, 7.0, 
12.0, 5.0, 18.0, 4.0, 139.0, 29.0, 61.0, 25.0, 24.0, 59.0, 0.0, 
7.0, 180.0, 0.0, 0.0, 0.0, 6.0, 619.0, 159.0, 1.0, 0.0, 1.0, 5.0, 
0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 19.0, 12.0, 0.0, 0.0, 59.0, 36.0, 
0.0, 0.0, 0.0, -0.05375744047619047, 0.4927827380952381, 32.0]

--->

array([[-1.4530485 , -0.03725366,  0.53727615,  0.51361116,  0.26511576,
         0.79677552,  0.01716853,  0.77574926,  0.24912955, -0.64725461,
         0.01852962, -0.02733052]])
'''
import sys, os, pickle, json
import numpy as np

# files 
problemtype=sys.argv[1]
picklefile=sys.argv[2]
jsonfile=picklefile[0:-7]+'.json'

# load the model 
os.chdir(problemtype+'_transformer')
g=pickle.load(open(picklefile,'rb'))
# load the corresponding SON
h=json.load(open(jsonfile))
# see sample input 
sample=h['sample input X']
# reshape the data to output array 
print('----------------------------------')
print('        TRANSFORMING DATA         ')
print('----------------------------------')
print(sample)
print('-->')
print(g.transform(np.array(sample).reshape(1,-1)))
