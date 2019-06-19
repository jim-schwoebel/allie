
from PIL import Image, ImageSequence
import os 
import vidaug.vidaug.augmentors as va

def gif_loader(path, modality="RGB"):
    frames = []
    with open(path, 'rb') as f:
        with Image.open(f) as video:
            index = 1
            for frame in ImageSequence.Iterator(video):
                frames.append(frame.convert(modality))
                index += 1
        return frames

file = 'sample.mp4'
# convert file to gif 
if file[-4:] != '.gif':
	# only take first 10 seconds.
	os.system('ffmpeg -i ./videos/%s ./videos/%s'%(file, file[0:-4]+'.gif'))
	file=file[0:-4]+'.gif'

frames = gif_loader(os.getcwd()+"/videos/%s"%(file))


sometimes = lambda aug: va.Sometimes(1, aug) # Used to apply augmentor with 100% probability

seq = va.Sequential([ # randomly rotates the video with a degree randomly choosen from [-10, 10]  
    sometimes(va.HorizontalFlip()) # horizontally flip the video with 100% probability
])

#augment the frames
video_aug = seq(frames)

# save augmentad frames as gif 
video_aug[0].save(file[0:-4]+'.gif', save_all=True, append_images=video_aug[1:], duration=100, loop=0)
os.system('ffmpeg -i '+file[0:-4]+'.gif '+file[0:-4]+'_enhanced.mp4')
print(len(video_aug))
