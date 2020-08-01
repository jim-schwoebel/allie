import os, sys, shutil

def augment_eda(textfile, basedir):

	arg_num= 1
	text='1\t'+open(os.getcwd()+'/'+textfile).read()
	textfile2=open(textfile,'w')
	textfile2.write(text)
	textfile2.close()
	
	shutil.copy(os.getcwd()+'/'+textfile,basedir+'/helpers/eda_nlp/data/'+textfile)

	newfile='augmented_'+textfile
	os.system('python3 %s/helpers/eda_nlp/code/augment.py --input=%s --output=%s --num_aug=%s --alpha=0.05'%(basedir, textfile, newfile, str(arg_num)))

	shutil.copy(basedir+'/helpers/eda_nlp/data/'+newfile, os.getcwd()+'/'+newfile)

	os.remove(basedir+'/helpers/eda_nlp/data/'+textfile)
	os.remove(basedir+'/helpers/eda_nlp/data/'+newfile)
	