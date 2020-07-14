'''
get audio from each file (for processing)
'''
import os

folder=input('what playlist do you want audio?')
os.chdir(folder)

listdir=os.listdir()

for i in range(len(listdir)):
    if listdir[i][-4:]=='.mp4':
        os.system('ffmpeg -i %s %s'%(listdir[i],listdir[i][0:-4]+'.wav'))
        
