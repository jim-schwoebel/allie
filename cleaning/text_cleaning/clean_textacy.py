'''
               AAA               lllllll lllllll   iiii                      
              A:::A              l:::::l l:::::l  i::::i                     
             A:::::A             l:::::l l:::::l   iiii                      
            A:::::::A            l:::::l l:::::l                             
           A:::::::::A            l::::l  l::::l iiiiiii     eeeeeeeeeeee    
          A:::::A:::::A           l::::l  l::::l i:::::i   ee::::::::::::ee  
         A:::::A A:::::A          l::::l  l::::l  i::::i  e::::::eeeee:::::ee
        A:::::A   A:::::A         l::::l  l::::l  i::::i e::::::e     e:::::e
       A:::::A     A:::::A        l::::l  l::::l  i::::i e:::::::eeeee::::::e
      A:::::AAAAAAAAA:::::A       l::::l  l::::l  i::::i e:::::::::::::::::e 
     A:::::::::::::::::::::A      l::::l  l::::l  i::::i e::::::eeeeeeeeeee  
    A:::::AAAAAAAAAAAAA:::::A     l::::l  l::::l  i::::i e:::::::e           
   A:::::A             A:::::A   l::::::ll::::::li::::::ie::::::::e          
  A:::::A               A:::::A  l::::::ll::::::li::::::i e::::::::eeeeeeee  
 A:::::A                 A:::::A l::::::ll::::::li::::::i  ee:::::::::::::e  
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee  


/  __ \ |                (_)              / _ \ | ___ \_   _|  _ 
| /  \/ | ___  __ _ _ __  _ _ __   __ _  / /_\ \| |_/ / | |   (_)
| |   | |/ _ \/ _` | '_ \| | '_ \ / _` | |  _  ||  __/  | |      
| \__/\ |  __/ (_| | | | | | | | | (_| | | | | || |    _| |_   _ 
 \____/_|\___|\__,_|_| |_|_|_| |_|\__, | \_| |_/\_|    \___/  (_)
                                   __/ |                         
                                  |___/                          
 _____         _   
|_   _|       | |  
  | | _____  _| |_ 
  | |/ _ \ \/ / __|
  | |  __/>  <| |_ 
  \_/\___/_/\_\\__|
                   
This script cleans folders of .TXT files with things like 
removing whitespace, normalizing hyphenized words, and many
other things using the textacy module: https://pypi.org/project/textacy/

This is enabled if the default_text_cleaners=['clean_textacy']
'''
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
	newfile='cleaned_'+textfile
	textfile2=open(newfile,'w')
	textfile2.write(text)
	textfile2.close()
	os.remove(textfile)

	return [newfile]