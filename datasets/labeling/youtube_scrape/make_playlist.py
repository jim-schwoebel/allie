'''
================================================ 
          YOUTUBE_SCRAPE REPOSITORY                     
================================================ 

repository name: youtube_scrape 
repository version: 1.0 
repository link: https://github.com/jim-schwoebel/youtube_scrape 
author: Jim Schwoebel 
author contact: js@neurolex.co 
description: Library for scraping youtube videos. Alternative to pafy, pytube, and youtube-dl. 
license category: opensource 
license: Apache 2.0 license 
organization name: NeuroLex Laboratories, Inc. 
location: Seattle, WA 
website: https://neurolex.ai 
release date: 2018-07-23 

This code (youtube_scrape) is hereby released under a Apache 2.0 license license. 

For more information, check out the license terms below. 

================================================ 
                LICENSE TERMS                      
================================================ 

Copyright 2018 NeuroLex Laboratories, Inc. 
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

     http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

================================================ 
                SERVICE STATEMENT                    
================================================ 

If you are using the code written for a larger project, we are 
happy to consult with you and help you with deployment. Our team 
has >10 world experts in kafka distributed architectures, microservices 
built on top of Node.JS / python / docker, and applying machine learning to 
model speech and text data. 

We have helped a wide variety of enterprises - small businesses, 
researchers, enterprises, and/or independent developers. 

If you would like to work with us let us know @ js@neurolex.co. 
'''
###########################################################################
#                       IMPORT STATEMENTS                                ##
###########################################################################
import requests, json, os 
from bs4 import BeautifulSoup
from pytube import YouTube

###########################################################################
#                       HELPER FUNCTIONS                                ##
###########################################################################
def scrapelinks(playlist, links):
    #https://www.youtube.com/playlist?list=PL1v-PVIZFDsqbzPIsEPZPnvcgIQ8bNTKS
    page=requests.get(playlist)
    base='https://www.youtube.com/watch?v='
    soup=BeautifulSoup(page.content, 'lxml')

    g=soup.find_all('tr',class_='pl-video yt-uix-tile ')
    entries=list()
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

    return entries, len(entries), totaltime, links

###########################################################################
##                          MAIN CODE BASE                               ##
###########################################################################

playlists=list()
entries=list()
t=1
totalnum=0
totaltime=0
links=list()

playlist_name=input('what do you want to name this playlist (e.g. angry)?')

while t>0:
    
    #try:

    playlist=input('what is the playlist id or URL?')
    if playlist.find('playlist?list=')>0:
        playlists.append(playlist)
        entry, enum, nowtime, link=scrapelinks(playlist, links)
        links=links+link 
        totalnum=totalnum+enum
        totaltime=totaltime+nowtime 
        entries=entries+entry
    elif playlist not in ['', 'n']:
        playlist='https://www.youtube.com/playlist?list='+playlist
        playlists.append(playlist)
        entry, enum, nowtime, link=scrapelinks(playlist, links)
        links=links+link 
        totalnum=totalnum+enum
        totaltime=totaltime+nowtime 
        entries=entries+entry
    else:
        break

    #except:

        #print('error') 

os.chdir(os.getcwd()+'/playlists')

data={
    'entrynum':totalnum,
    'total time':totaltime,
    'playlist url':playlists,
    'entries':entries,
}

jsonfile=open(playlist_name+'.json','w')
json.dump(data,jsonfile)
jsonfile.close()
