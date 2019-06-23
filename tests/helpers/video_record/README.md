# video_record
This is a repository for recording videos alongside screen information on mac computers. Audio, video, keyboard typing, mouse clicks, and computer screens are all recorded in parallel, and a video is created from the captured webcam/microphone (with audio/video streams connected) and the computer screen. 

![](https://media.giphy.com/media/3tGUlL7jKLhkupHvQa/giphy.gif)

## Why I created this repository

I created this script because I found few solutions that could seamlessly connect audio, video, and screen information written in Python. Most of the repositories out there record only one stream like an audio stream (e.g. [sounddevice](https://python-sounddevice.readthedocs.io/en/0.3.12/)) or video stream (e.g. [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)). Moreover, this seemed like a difficult task because if you recorded many of these streams in parallel the timing may be off with the lips to the audio if you merged the two recordings.

Therefore, I looked deeper at the problem and it seemed like you could use Redis (using [ray](https://github.com/ray-project/ray)) or a similar distributed cue to record parallel instances of all the channels desired (audio, video, keyboard typing, mouse clicks, and computer screen recordings). Then, you could use packages like FFmpeg to merge the various channels. You can also use custom thresholding to make sure the audio and video are aligned. 

## Live demo

Click on the image below to see a live demo of a recorded screencast! 

![](https://github.com/jim-schwoebel/video_record/blob/master/screenshot.png)

## Getting started

To get started, you can create a virtual environment and install all dependencies.
```
git clone git@github.com:jim-schwoebel/video_record.git
cd video_record
python3 -m venv video_record
source video_record/bin/activate
pip3 install -r requirements.txt 
```

Now you can run main script. There are two arguments - the file name ('test.avi') and the number of seconds that you wish to record (60):

```
python3 record.py test.avi 60 /Users/jimschwoebel/downloads/
```
This will record a 60 second video file and transfer the file to the /downloads directory.

## Feedback
Any feedback this repository is greatly appreciated. 
* If you find something that is missing or doesn't work, please consider opening a [GitHub issue](https://github.com/jim-schwoebel/video_record/issues).
* If you want to learn more about voice computing, check out [Voice Computing in Python](https://github.com/jim-schwoebel/voicebook) book.
* If you'd like to be mentored by someone on our team, check out the [Innovation Fellows Program](http://neurolex.ai/research).
* If you want to talk to me directly, please send me an email @ js@neurolex.co. 

## License
This repository is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). 

## resources
* [ffmpeg](https://ffmpeg.org/)
* [openCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)
* [sounddevice](https://python-sounddevice.readthedocs.io/en/0.3.12/)
