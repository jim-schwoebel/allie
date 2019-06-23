import sys, os, ffmpeg_normalize 
'''
go to the proper folder from cmd line
'''

def normalize_volume(wavfile):
    # using peak normalization method 
    os.system('ffmpeg-normalize %s -nt peak -t 0 -o peak_normalized.wav'%(wavfile))
    os.remove(wavfile)
    os.rename('peak_normalized.wav', wavfile)

try:
    foldername=sys.argv[1]
    os.chdir(foldername)
except:
    foldername=input('what folder would you like to normalize audio file volumes?')
    directory=os.getcwd()
    os.chdir(directory+'/%s'%(foldername))

# remove duplicates using filecmp
listdir=os.listdir()
normalized_files=list()

# in the very improbable chance there are file naming conflicts, let's add this line of code
if 'peak_normalized.wav' in listdir:
    os.rename('peak_normalized.wav', 'peak_normalized_2.wav')

print('-----------------------------')
print('   NORMALIZING VOLUMES ...   ')
print('-----------------------------')

for i in range(len(listdir)):
    if listdir[i][-4:]=='.wav':
        normalize_volume(listdir[i])
        normalized_files.append(listdir[i])
 
print('normalizing the audio files below')
print(normalized_files)
