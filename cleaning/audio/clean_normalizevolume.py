import sys, os, ffmpeg_normalize 
'''
go to the proper folder from cmd line
'''

def clean_normalizevolume(audiofile):
    # using peak normalization method 
    os.system('ffmpeg-normalize %s -nt peak -t 0 -o peak_normalized.wav'%(audiofile))
    os.remove(audiofile)
    os.rename('peak_normalized.wav', audiofile)