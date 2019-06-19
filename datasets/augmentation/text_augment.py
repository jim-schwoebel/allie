import os, sys

path_to_input=sys.argv[1]
path_to_output=sys.argv[2]
arg_num= sys.argv[3]

os.system('python3 ./eda_nlp/code/augment.py --input=./data/text/%s --output=./data/text/%s --num_aug=%s --alpha=0.05'%(path_to_input, path_to_output, arg_num))
