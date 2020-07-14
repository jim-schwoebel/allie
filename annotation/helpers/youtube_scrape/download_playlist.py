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

================================================ 
                    NOTE                  
================================================ 
Download a playlist from the URLS previously generated
with make_playlist.py script.

Make sure you have at least around 10GB of disk space
before bulk downloading videos, as they can take up a lot of space.

'''
import requests, json, os, shutil
from bs4 import BeautifulSoup
from pytube import YouTube

playlist_name=input('what is the name of the playlist to download?')
hostdir=os.getcwd()
os.chdir(os.getcwd()+'/playlists/')

try:
    if playlist_name[-5:] != '.json':
        g=json.load(open(playlist_name+'.json'))
        entries=g['entries']
        links=list()
    elif playlist_name[-5:] == '.json':
        g=json.load(open(playlist_name))
        entries=g['entries']
        links=list() 
except:
    print('error loading playlist. Please make sure it is in the playlists folder and you type in the name properly. \n\n For example yc_podcast.json ==> yc_podcast or yc_podcast.json')

if playlist_name[-5:]=='.json':
    foldername=playlist_name[0:-5]
else:
    foldername=playlist_name
    foldername
try:
    os.mkdir(foldername)
    os.chdir(foldername)
except:
    shutil.rmtree(foldername)
    os.mkdir(foldername)
    
    os.chdir(foldername)

for i in range(len(entries)):
    link=entries[i]['link']
    links.append(link)
    print(link)

# download files 
for i in range(len(links)):
    try:
        link=links[i]
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

# now make audio for each .mp4 file 
listdir=os.listdir()

for i in range(len(listdir)):
    if listdir[i][-4:]=='.mp4':
        os.system('ffmpeg -i %s %s'%(listdir[i],listdir[i][0:-4]+'.wav'))
