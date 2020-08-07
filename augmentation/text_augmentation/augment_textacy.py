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
  


 / _ \                                 | |      | | (_)            
/ /_\ \_   _  __ _ _ __ ___   ___ _ __ | |_ __ _| |_ _  ___  _ __  
|  _  | | | |/ _` | '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \ 
| | | | |_| | (_| | | | | | |  __/ | | | || (_| | |_| | (_) | | | |
\_| |_/\__,_|\__, |_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|
        __/ |                                                
       |___/                                                 
  ___  ______ _____       _____         _   
 / _ \ | ___ \_   _|  _  |_   _|       | |  
/ /_\ \| |_/ / | |   (_)   | | _____  _| |_ 
|  _  ||  __/  | |         | |/ _ \ \/ / __|
| | | || |    _| |_   _    | |  __/>  <| |_ 
\_| |_/\_|    \___/  (_)   \_/\___/_/\_\\__|
'''
import os, sys, shutil, textacy
import textacy.augmentation.transforms as transforms

def augment_textacy(textfile, basedir):
  filename1=textfile
  text = open(textfile).read() 
  # "The quick brown fox jumps over the lazy dog."
  doc = textacy.make_spacy_doc(text, lang="en")
  tfs = [transforms.substitute_word_synonyms, transforms.delete_words, transforms.swap_chars, transforms.delete_chars]
  augmenter=textacy.augmentation.augmenter.Augmenter(tfs, num=[0.5, 0.5, 0.5, 0.5])
  augmented_text=augmenter.apply_transforms(doc)
  filename2='augmented_'+textfile
  textfile=open(filename2,'w')
  textfile.write(str(augmented_text))
  textfile.close()
  return [filename1, filename2]