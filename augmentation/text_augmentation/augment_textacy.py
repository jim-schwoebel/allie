import os, sys, shutil, textacy
import textacy.augmentation.transforms as transforms

def augment_textacy(textfile, basedir):

	text = open(textfile).read() 
	# "The quick brown fox jumps over the lazy dog."
	doc = textacy.make_spacy_doc(text, lang="en")
	tfs = [transforms.substitute_word_synonyms, transforms.delete_words, transforms.swap_chars, transforms.delete_chars]
	augmenter=textacy.augmentation.augmenter.Augmenter(tfs, num=[0.5, 0.5, 0.5, 0.5])
	augmented_text=augmenter.apply_transforms(doc)

	textfile=open('augmented_'+textfile,'w')
	textfile.write(str(augmented_text))
	textfile.close()


	