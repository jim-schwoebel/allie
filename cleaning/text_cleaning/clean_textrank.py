import os, uuid, sys

try:
	import textrank
except:
	os.system('pip3 install git+git://github.com/davidadamojr/TextRank.git')

def extract_summary(textfile):
	summary=str(uuid.uuid4())+'.txt'
	os.system('textrank extract-summary %s > %s'%(textfile, summary))
	summary=open(summary).read()
	return summary

def extract_phrases(textfile):
	phrases=str(uuid.uuid4())+'.txt'
	os.system('textrank extract-phrases %s > %s'%(textfile, phrases))
	phrases=open(phrases).read()
	return phrases

# phrases = extract_phrases('test.txt')
# summary = extract_summary('test.txt')
