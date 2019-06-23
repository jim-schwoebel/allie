import sys, os, cv2, time, readchar, datetime
import soundfile as sf 
import sounddevice as sd 
import numpy as np
from multiprocessing import Process
import ray, json
import subprocess32 as sp
import pyautogui, shutil, zipfile 
from natsort import natsorted

curdir=os.getcwd()
try:
	os.mkdir('temp')
	os.chdir('temp')
except:
	shutil.rmtree('temp')
	os.mkdir('temp')
	os.chdir('temp')

def zip(src, dst):
    zf = zipfile.ZipFile("%s.zip"%(dst), "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            arcname = absname[len(abs_src) + 1:]
            print('zipping %s as %s'%(os.path.join(dirname, filename),arcname))
            zf.write(absname, arcname)
    zf.close()

def calc_duration(vid_file_path):
    ''' Video's duration in seconds, return a float number
    '''
    def probe(vid_file_path):
	    ''' Give a json from ffprobe command line

	    @vid_file_path : The absolute (full) path of the video file, string.
	    '''
	    if type(vid_file_path) != str:
	        raise Exception('Gvie ffprobe a full file path of the video')
	        return

	    command = ["ffprobe",
	            "-loglevel",  "quiet",
	            "-print_format", "json",
	             "-show_format",
	             "-show_streams",
	             vid_file_path
	             ]

	    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)
	    out, err = pipe.communicate()
	    return json.loads(out)


    _json = probe(vid_file_path)

    if 'format' in _json:
        if 'duration' in _json['format']:
            return float(_json['format']['duration'])

    if 'streams' in _json:
        # commonly stream 0 is the video
        for s in _json['streams']:
            if 'duration' in s:
                return float(s['duration'])

    # if everything didn't happen,
    # we got here because no single 'return' in the above happen.
    raise Exception('I found no duration')
    #return None

@ray.remote
def mouse(filename, duration):
	print('recording mouse movements')
	print('--> mouse_%s.json'%(filename[0:-4]))
	# get features from the mouse to detect activity
	deltat=.1
	pyautogui.PAUSE = 1
	pyautogui.FAILSAFE = True
	positions=list()

	#log 20 mouse movements
	for i in range(0, duration):
	    curpos=pyautogui.position()
	    positions.append(curpos)

	jsonfile=open('mouse_%s.json'%(filename[0:-4]),'w')
	data={'mouse_positions':positions}
	json.dump(data,jsonfile)
	jsonfile.close()

	return positions

@ray.remote
def keyboard(filename, duration):
	print('recording keyboard')
	print('--> keyboard_%s'%(filename[0:-4]))

	# capture keyboard features 
	def getch():
		ch=readchar.readkey()
		return ch

	charlist=list()
	start=time.time()
	end=time.time()

	while end-start<duration:
		end=time.time()
		charlist.append(getch())

	total_time=end-start
	if total_time > duration+15:
		# allow 15 second window after for typing activity 
		pass
	else:
		jsonfile=open('keyboard_%s.json'%(filename[0:-4]),'w')
		data={'keyboard':positions}
		json.dump(data,jsonfile)
		jsonfile.close()

@ray.remote 
def screen_record(filename, duration):
	curdir=os.getcwd()
	# function to write a video from an image for duration X
	def video_write(imagename, videoname, duration2):
		img=cv2.imread(imagename)
		height, width, layers = img.shape 
		video=cv2.VideoWriter(videoname, -1, 1, (width, height))

		for i in range(duration2):
			video.write(cv2.imread(imagename))

		cv2.destroyAllWindows()
		video.release() 

	# 1 screen per second 
	newfilename=filename[0:-4]+'_screenshots.avi'
	print('making screenshots (.AVI)')
	print('--> screenshots.avi')
	foldername=filename[0:-4]+'_screenshots'
	try:
		os.mkdir(foldername)
		os.chdir(foldername)
	except:
		shutil.rmtree(foldername)
		os.mkdir(foldername)
		os.chdir(foldername)
	
	start_time=time.time()
	count=0
	while True:
		if time.time()-start_time < duration:
			pyautogui.screenshot(str(count)+".png")
			count=count+1
		else:
			break

	# this will record 1 screenshot per time 
	files=natsorted(os.listdir())
	print(files)
	# now make a video from all the individual screenshots 
	for i in range(len(files)):
		print('making '+files[i][0:-4]+'.avi')
		video_write(files[i], files[i][0:-4]+'.avi', 1)
		# os.remove(files[i])

	# now combine all videos 
	files=natsorted(os.listdir())
	file=open('mylist.txt','w')
	for i in range(len(files)):
		if files[i][-4:]=='.avi':
			file.write("file '%s' \n"%(files[i]))
	file.close()

	command='ffmpeg -f concat -i mylist.txt -c copy %s'%(newfilename)
	os.system(command)
	# convert to .mp4 format 
	os.system('ffmpeg -i %s %s'%(newfilename, newfilename[0:-4]+'.mp4'))
	os.remove(newfilename)

	vid_duration=calc_duration(newfilename[0:-4]+'.mp4')
	speedfactor=duration/vid_duration
	print(speedfactor)
	os.system('ffmpeg -i %s -filter:v "setpts=%s*PTS" %s'%(newfilename[0:-4]+'.mp4', str(speedfactor), newfilename[0:-4]+'_2.mp4'))
	os.remove(newfilename[0:-4]+'.mp4')
	os.rename(newfilename[0:-4]+'_2.mp4',newfilename[0:-4]+'.mp4')
	shutil.move(os.getcwd()+'/'+newfilename[0:-4]+'.mp4', curdir+'/'+newfilename[0:-4]+'.mp4')

@ray.remote
def video_record(filename, duration):
	print('recording video (.AVI)')
	print('--> '+filename)

	t0 = time.time() # start time in seconds

	video=cv2.VideoCapture(0)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')

	frame_width = int(video.get(3))
	frame_height = int(video.get(4))
	out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

	a=0
	start=time.time()

	while True:
		a=a+1
		check, frame=video.read()
		#print(check)
		#print(frame)
		gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		out.write(frame)
		#cv2.imshow("frame",gray)
		end=time.time()
		if end-start>duration:
		    break 

	print(a)
	video.release()
	out.release() 
	cv2.destroyAllWindows()

@ray.remote 
def audio_record(filename, duration):
	print('recording audio (.WAV)')
	print('--> '+filename)
	time.sleep(0.50)
	fs=44100
	channels=2
	myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
	sd.wait()
	sf.write(filename, myrecording, fs)

def video_audio_record(videofile, duration):
	# record each in parallel 
	# runInParallel(audio_record(filename[0:-4]+'.wav', duration), video_record(filename,duration))
	audiofile=filename[0:-4]+'.wav'
	ray.get([video_record.remote(videofile,duration), audio_record.remote(audiofile, duration), screen_record.remote(filename, duration), mouse.remote(filename, duration)])
	#os.system('ffmpeg -i %s -i %s -c:v copy -c:a aac -strict experimental %s'%(videofile, audiofile, videofile))
	#os.remove(audiofile)
	# connect two files 

filename=sys.argv[1]
duration=int(sys.argv[2])
train_dir=sys.argv[3]

print(filename)
print(duration)

if filename.find('.avi') > 0:
	ray.init()
	video_audio_record(filename, duration)

# for testing !! (calculate the right framerate for duration)
vid_duration=calc_duration(os.getcwd()+'/'+filename)
print(vid_duration)

# initialize names of stuff 
audiofile=filename[0:-4]+'.wav'
newfilename=filename[0:-4]+'_new.mp4'
newfilename2=filename[0:-4]+'_new2.mp4'

if vid_duration > duration or vid_duration < duration:
	# convert to be proper length
	print('converting to 20 seconds of video...')
	# following ffmpeg documentation https://trac.ffmpeg.org/wiki/How%20to%20speed%20up%20/%20slow%20down%20a%20video
	speedfactor=duration/vid_duration
	print(speedfactor)
	os.system('ffmpeg -i %s -filter:v "setpts=%s*PTS" %s'%(filename, str(speedfactor), newfilename))
	os.system('ffmpeg -i %s -i %s -c:v copy -c:a aac -strict experimental %s'%(newfilename, audiofile, newfilename2))
	#os.remove(filename)
	#os.remove(newfilename)
	#os.rename(newfilename2, filename)
else:
	os.system('ffmpeg -i %s -i %s -c:v copy -c:a aac -strict experimental %s'%(filename, audiofile, newfilename2))
	#os.remove(filename)
	#os.rename(newfilename2, filename)


# make everything into one video 

one=newfilename2[0:-4]+'_.mp4'
two=filename[0:-4]+'_screenshots_2.mp4'

#resize video 1 
os.system('ffmpeg -i %s -vf scale=640:360 %s -hide_banner'%(newfilename2, one))
# resize video 2 
os.system('ffmpeg -i %s -vf scale=640:360 %s -hide_banner'%(filename[0:-4]+'_screenshots.mp4', two))

# combine 
os.system('ffmpeg -i %s -i %s -filter_complex hstack output.mp4'%(one, two))
#os.system('open output.mp4')

# remove temp files and rename
os.remove(one)
os.remove(two)
os.remove(filename)
os.rename(newfilename, filename[0:-4]+'.mp4')
os.remove(filename[0:-4]+'.mp4')
os.rename(newfilename2, filename[0:-4]+'.mp4')
shutil.rmtree(filename[0:-4]+'_screenshots')

os.chdir(curdir)
file_dir=os.getcwd()+'/'+filename[0:-4]
shutil.move(file_dir+'/'+filename[0:-4]+'.mp4', train_dir+'/'+filename[0:-4]+'.mp4')

shutil.rmtree('temp')
shutil.rmtree(filename[0:-4])
