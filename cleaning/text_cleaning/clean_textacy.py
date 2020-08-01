import os
import textacy.preprocessing as preprocessing

def clean_textacy(textfile):
	text=open(textfile).read()
	text=preprocessing.normalize_whitespace(text)
	text=preprocessing.normalize.normalize_hyphenated_words(text)
	text=preprocessing.normalize.normalize_quotation_marks(text)
	text=preprocessing.normalize.normalize_unicode(text)
	text=preprocessing.remove.remove_accents(text)
	# text=preprocessing.remove.remove_punctuation(text)
	text=preprocessing.replace.replace_currency_symbols(text)
	text=preprocessing.replace.replace_emails(text)
	text=preprocessing.replace.replace_hashtags(text)
	# text=preprocessing.replace.replace_numbers(text)
	text=preprocessing.replace.replace_phone_numbers(text)
	text=preprocessing.replace.replace_urls(text)
	text=preprocessing.replace.replace_user_handles(text)

	print(text)
	# now replace the original doc with cleaned version
	textfile2=open('cleaned_'+textfile,'w')
	textfile2.write(text)
	textfile2.close()

	os.remove(textfile)
	
# clean_textacy('test.txt')