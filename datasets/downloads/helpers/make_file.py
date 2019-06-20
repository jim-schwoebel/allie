import os, json

def make_jsonfile(textfile): 
	g=open(textfile).read()
	h=g.split('*')
	os.chdir(textfile[0:-3])
	for i in range(len(h)):
		i1=h[i].find('[')
		i2=h[i].find(']')
		i3=h[i].find('(')
		i4=h[i].find(')')
		i5=h[i].find('-')
		name=h[i][i1+1:i2]
		print(name)
		description=h[i][i5:].replace('\n','')
		link=h[i][i3+1:i4]
		data={name: {'description': description,
					'link': link}}
		jsonfile=open(name.replace(' ','_')+'.json','w')
		json.dump(data,jsonfile)
		jsonfile.close()

text=input('what file would you like to featurize?')
make_jsonfile(text)