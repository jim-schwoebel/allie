import os, uuid, sys

try:
	import textrank
except:
	os.system('pip3 install git+git://github.com/davidadamojr/TextRank.git')

def clean_summary(textfile):
	summary='summary_'+textfile
	os.system('textrank extract-summary %s > %s'%(textfile, summary))
	summary=open(summary).read()
	os.remove(textfile)
