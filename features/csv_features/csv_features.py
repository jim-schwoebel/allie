import pandas as pd 
import nltk, json, os 
from nltk import word_tokenize 
import numpy as np
from textblob import TextBlob

def nltk_featurize(transcript):

	#alphabetical features 
	a=transcript.count('a')
	b=transcript.count('b')
	c=transcript.count('c')
	d=transcript.count('d')
	e=transcript.count('e')
	f=transcript.count('f')
	g_=transcript.count('g')
	h=transcript.count('h')
	i=transcript.count('i')
	j=transcript.count('j')
	k=transcript.count('k')
	l=transcript.count('l')
	m=transcript.count('m')
	n=transcript.count('n')
	o=transcript.count('o')
	p=transcript.count('p')
	q=transcript.count('q')
	r=transcript.count('r')
	s=transcript.count('s')
	t=transcript.count('t')
	u=transcript.count('u')
	v=transcript.count('v')
	w=transcript.count('w')
	x=transcript.count('x')
	y=transcript.count('y')
	z=transcript.count('z')
	atsymbol=transcript.count('@')
	space=transcript.count(' ')

	#numerical features and capital letters 
	num1=transcript.count('0')+transcript.count('1')+transcript.count('2')+transcript.count('3')+transcript.count('4')+transcript.count('5')+transcript.count('6')+transcript.count('7')+transcript.count('8')+transcript.count('9')
	num2=transcript.count('zero')+transcript.count('one')+transcript.count('two')+transcript.count('three')+transcript.count('four')+transcript.count('five')+transcript.count('six')+transcript.count('seven')+transcript.count('eight')+transcript.count('nine')+transcript.count('ten')
	number=num1+num2
	capletter=sum(1 for c in transcript if c.isupper())

	#part of speech 
	text=word_tokenize(transcript)
	g=nltk.pos_tag(transcript)
	cc=0
	cd=0
	dt=0
	ex=0
	in_=0
	jj=0
	jjr=0
	jjs=0
	ls=0
	md=0
	nn=0
	nnp=0
	nns=0
	pdt=0
	pos=0
	prp=0
	prp2=0
	rb=0
	rbr=0
	rbs=0
	rp=0
	to=0
	uh=0
	vb=0
	vbd=0
	vbg=0
	vbn=0
	vbp=0
	vbp=0
	vbz=0
	wdt=0
	wp=0
	wrb=0

	for i in range(len(g)):
		if g[i][1] == 'CC':
			cc=cc+1
		elif g[i][1] == 'CD':
			cd=cd+1
		elif g[i][1] == 'DT':
			dt=dt+1
		elif g[i][1] == 'EX':
			ex=ex+1
		elif g[i][1] == 'IN':
			in_=in_+1
		elif g[i][1] == 'JJ':
			jj=jj+1
		elif g[i][1] == 'JJR':
			jjr=jjr+1                   
		elif g[i][1] == 'JJS':
			jjs=jjs+1
		elif g[i][1] == 'LS':
			ls=ls+1
		elif g[i][1] == 'MD':
			md=md+1
		elif g[i][1] == 'NN':
			nn=nn+1
		elif g[i][1] == 'NNP':
			nnp=nnp+1
		elif g[i][1] == 'NNS':
			nns=nns+1
		elif g[i][1] == 'PDT':
			pdt=pdt+1
		elif g[i][1] == 'POS':
			pos=pos+1
		elif g[i][1] == 'PRP':
			prp=prp+1
		elif g[i][1] == 'PRP$':
			prp2=prp2+1
		elif g[i][1] == 'RB':
			rb=rb+1
		elif g[i][1] == 'RBR':
			rbr=rbr+1
		elif g[i][1] == 'RBS':
			rbs=rbs+1
		elif g[i][1] == 'RP':
			rp=rp+1
		elif g[i][1] == 'TO':
			to=to+1
		elif g[i][1] == 'UH':
			uh=uh+1
		elif g[i][1] == 'VB':
			vb=vb+1
		elif g[i][1] == 'VBD':
			vbd=vbd+1
		elif g[i][1] == 'VBG':
			vbg=vbg+1
		elif g[i][1] == 'VBN':
			vbn=vbn+1
		elif g[i][1] == 'VBP':
			vbp=vbp+1
		elif g[i][1] == 'VBZ':
			vbz=vbz+1
		elif g[i][1] == 'WDT':
			wdt=wdt+1
		elif g[i][1] == 'WP':
			wp=wp+1
		elif g[i][1] == 'WRB':
			wrb=wrb+1		

	#sentiment
	tblob=TextBlob(transcript)
	polarity=float(tblob.sentiment[0])
	subjectivity=float(tblob.sentiment[1])

	#word repeats
	words=transcript.split()
	newlist=transcript.split()
	repeat=0
	for i in range(len(words)):
		newlist.remove(words[i])
		if words[i] in newlist:
			repeat=repeat+1 

	features=np.array([a,b,c,d,
	e,f,g_,h,
	i,j,k,l,
	m,n,o,p,
	q,r,s,t,
	u,v,w,x,
	y,z,atsymbol,space,number,
	capletter,cc,cd,dt,
	ex,in_,jj,jjr,
	jjs,ls,md,nn,
	nnp,nns,pdt,pos,
	prp,prp2,rbr,rbs,
	rp,to,uh,vb,
	vbd,vbg,vbn,vbp,
	vbz,wdt,wp,wrb,
	polarity,subjectivity,repeat])

	labels=['a', 'b', 'c', 'd',
			'e','f','g','h',
			'i', 'j', 'k', 'l',
			'm','n','o', 'p',
			'q','r','s','t',
			'u','v','w','x',
			'y','z','atsymbol','space', 'numbers',
			'capletters','cc','cd','dt',
			'ex','in','jj','jjr',
			'jjs','ls','md','nn',
			'nnp','nns','pdt','pos',
			'prp','prp2','rbr','rbs',
			'rp','to','uh','vb',
			'vbd','vbg','vbn','vbp',
			'vbz', 'wdt', 'wp','wrb',
			'polarity', 'subjectivity','repeat']

	return features, labels

def get_categories(sample_list):
	tlist=list()
	for i in range(len(sample_list)):
		if sample_list[i] not in tlist:
			tlist.append(sample_list[i])

	# tdict=list()
	# for i in range(len(tlist)):
	# 	tdict[tlist[i]]=tlist[i]

	return tlist

def csv_featurize(csv_file, cur_dir):
	'''
	Take in a .CSV file and output
	numerical and categorical features 
	for analysis.
	'''

	os.chdir(cur_dir)
	g=pd.read_csv(csv_file)
	labels=list(g)
	g_=list()

	# only include features that are fully here with no missing values. 
	for i in range(len(g)):
		entry=list(g.iloc[i,:])
		skip=False 
		for j in range(len(entry)):
			if str(entry[j]) == 'nan':
				skip=True

		if skip == True:
			pass
		else:
			g_.append(entry)

	print(len(g_))
	
	# now we need to classify each as categorical or numerical data 
	output=g_[0]
	types=list()

	# determine uniqueness of each column (if <10, treat as categorical data, otherwise numeric data)
	masterlist=list()
	not_unique=list()
	unique=list()
	for i in range(len(g_)):
		for j in range(len(g_[i])):
			if g_[i][j] in masterlist:
				not_unique.append(labels[j])
			else:
				masterlist.append(g_[i][j])
				unique.append(labels[j])

	# now figure out uniqueness level of each label 
	labeltypes=list()
	for i in range(len(labels)):
		if not_unique.count(labels[i]) >= unique.count(labels[i]):
			# categorical 
			labeltype='categorical'
		else:
			labeltype='numerical'

		labeltypes.append(labeltype)

	#print(labels)
	#print(labeltypes)

	# Now we need to convert the .CSV file to .JSON labels 

	for i in range(len(g_)):
		# calculate features for entire CSV 
		features=np.array([])
		labels_=list() 
		for j in range(len(g_[i])):
			# if it's categorical, we need to create numbers for the categories and output the numbers here with a mapping 
			if labeltypes[j] == 'categorical':
				tlist =get_categories(list(g.iloc[:,j]))
				print(g_[i][j])
				print(tlist)
				features=np.append(features,np.array(tlist.index(g_[i][j])))
				labels_=labels_+[labeltypes[j]]

			# if it's a text string | numerical, we need to characterize the string with features, added NLTK feautrize here.
			elif labeltypes[j] == 'numerical':
				try:
					# try to make into an integer or float if fail them assume string
					value=float(g_[i][j])
					tlabel=labels[j]
					labels_=labels_+[tlabel+'_float']
				except:
					tfeatures, tlabels=nltk_featurize(labeltypes[j])
					features=np.append(features, np.array(tfeatures))
					tlabels2=list()
					for k in range(len(tlabels)):
						tlabels2.append(labels[j].replace(' ','_')+'_'+tlabels[k])

					labels_=labels_+tlabels2 

		# feature array per sample (output as .JSON)
		filename=str(i)+'.json'
		data={'features': features.tolist(),
			  'labels': labels_}
		jsonfile=open(filename, 'w')
		json.dump(data,jsonfile)
		jsonfile.close()
	
# csv_featurize('test.csv', os.getcwd()+'/test')