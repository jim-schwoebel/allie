import helpers.vidaug.vidaug.augmentors as va
from PIL import Image, ImageSequence
import os 
import moviepy.editor as mp

def augment_video(videofile, basedir):

    def gif_loader(path, modality="RGB"):
        frames = []
        with open(path, 'rb') as f:
            with Image.open(f) as video:
                index = 1
                for frame in ImageSequence.Iterator(video):
                    frames.append(frame.convert(modality))
                    index += 1
            return frames

    file = videofile
    # convert file to gif 
    if file[-4:] != '.gif':
    	# only take first 10 seconds.
    	os.system('ffmpeg -i %s %s'%(file, file[0:-4]+'.gif'))
    	file=file[0:-4]+'.gif'

    frames = gif_loader(os.getcwd()+"/%s"%(file))

    sometimes = lambda aug: va.Sometimes(0.75, aug) # Used to apply augmentor with 75% probability
    seq = va.Sequential([
        va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
        va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
        sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
    ])

    #augment the frames
    video_aug = seq(frames)

    # save augmentad frames as gif 
    video_aug[0].save(file[0:-4]+'.gif', save_all=True, append_images=video_aug[1:], duration=100, loop=0)

    clip = mp.VideoFileClip(file[0:-4]+'.gif')
    clip.write_videofile('augmented_'+file[0:-4]+'.mp4')

    os.remove(file[0:-4]+'.gif')

    print(len(video_aug))
