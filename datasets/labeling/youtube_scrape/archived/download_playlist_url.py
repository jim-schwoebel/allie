'''
Extract playlist URLs
(for further processing)
'''
import requests, json, os 
from bs4 import BeautifulSoup
from pytube import YouTube

base='https://www.youtube.com/watch?v='

playlist_name=input('what do you want to name this playlist (e.g. angry)?')
#angry
playlist=input('what is the playlist url?')
#https://www.youtube.com/playlist?list=PL1v-PVIZFDsqbzPIsEPZPnvcgIQ8bNTKS
page=requests.get(playlist)
soup=BeautifulSoup(page.content, 'lxml')

g=soup.find_all('tr',class_='pl-video yt-uix-tile ')
entries=list()
links=list()
totaltime=0

for i in range(len(g)):
    try:
        h=str(g[i])
        
        # get titles
        h1=h.find('data-title="')+len('data-title="')
        h2=h[h1:].find('"')
        title=h[h1:h1+h2]

        # get links
        h3=h.find('data-video-id="')+len('data-video-id="')
        h4=h[h3:].find('"')
        link=base+h[h3:h3+h4]

        # get duration (in seconds)
        h5=h.find('<div class="timestamp"><span aria-label="')
        h6=h[h5:]
        hsoup=BeautifulSoup(h6,'lxml')
        htext=hsoup.text.replace('\n','').replace(' ','')
        hmin=htext.split(':')
        duration=int(hmin[0])*60+int(hmin[1])
        totaltime=totaltime+duration

        if link not in links:

            # avoids duplicate links 
            links.append(link)

            entry={
                'title':title,
                'link':link,
                'duration':duration
                }
            
            entries.append(entry)

    except:
        print('error')

os.mkdir(playlist_name)
os.chdir(os.getcwd()+'/'+playlist_name)

data={
    'entrynum':len(entries),
    'total time':totaltime,
    'playlist url':playlist,
    'entries':entries,
}

jsonfile=open('entries.json','w')
json.dump(data,jsonfile)
jsonfile.close()

for i in range(len(entries)):
    try:
        link=entries[i]['link']
        print('downloading %s'%(link))
        YouTube(link).streams.first().download()
    except:
        print('error')

# rename videos in order
listdir=os.listdir()
for i in range(len(listdir)):
    if listdir[i][-5:] in ['.webm']:
        os.rename(listdir[i],str(i)+'.webm')
        os.system('ffmpeg -i %s %s'%(str(i)+'.webm',str(i)+'.mp4'))
        os.remove(str(i)+'.webm')
    elif listdir[i][-4:] in ['.mp4']:
        os.rename(listdir[i],str(i)+'.mp4')
    
