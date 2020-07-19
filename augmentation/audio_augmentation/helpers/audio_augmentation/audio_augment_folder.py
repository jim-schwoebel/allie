import os, sys, librosa, shutil, time, random, math
from pydub import AudioSegment
import soundfile as sf  

def remove_json():
	listdir=os.listdir()
	for i in range(len(listdir)):
		if listdir[i][-5:]=='.json':
			os.remove(listdir[i])
			
def convert_audio():
	listdir=os.listdir()
	for i in range(len(listdir)):
		if listdir[i][-4:]!='.wav':
			os.system('ffmpeg -i %s %s'%(listdir[i], listdir[i][0:-4]+'.wav'))
			os.remove(listdir[i])

def augsort(filename,augmented_files, directory):

	dirname=directory.split('/')[-1]
	curdir=os.getcwd()
	os.chdir(directory)
	newdir=os.getcwd()
	listdir=os.listdir()
	file=filename.split('.wav')[0]

	for i in range(len(augmented_files)):
		print(augmented_files[i])
		print(file)
		foldername=augmented_files[i].split(file+'_')
		print(foldername)
		foldername=foldername[1].replace('.wav','')

		if dirname+'_'+foldername not in listdir: 
			os.mkdir(dirname+'_'+foldername)
		shutil.move(newdir+'/'+augmented_files[i], newdir+'/'+dirname+'_'+foldername+'/'+augmented_files[i])

	os.chdir(curdir)

def augment_dataset(filename, opusdir, curdir):

	def microphone_adjust(filename):
		'''
		takes in an audio file and filters it to different microphone
		configurations (sample rates). Perhaps change sample rate here?
		'''

		return ''
		
	def normalize_volume(filename):

		'''
		takes in an audio file and outputs files normalized to 
		different volumes. This corrects for microphone distance and ages.

		Note that in using ffmpeg-normalize, this mimicks real world-use.
		An alternative could be to use SoX to move up or down volume.
		'''

		def change_volume(filename, vol):
		    # rename file
		    if vol > 1:
		        new_file=filename[0:-4]+'_increase_vol.wav'
		    else:
		        new_file=filename[0:-4]+'_decrease_vol.wav'

		    # changes volume, vol, by input 
		    os.system('sox -v %s %s %s'%(str(vol),filename,new_file))

		    return new_file 

		filenames=list()
		basefile=filename[0:-4]
		# using peak normalization
		os.system('ffmpeg-normalize %s -nt peak -t 0 -o %s_peak_normalized.wav'%(filename, basefile))
		filenames.append('%s_peak_normalized.wav'%(basefile))
		
		# using low volume 
		# os.system('ffmpeg-normalize %s low.wav -o file1-normalized.wav -o %s-normalized_1.wav'%(filename, basefile))
		# filenames.append('%s-normalized_1.wav'%(basefile))
		
		# using moderate volume
		# os.system('ffmpeg-normalize %s moderate.wav -o file3-normalized.wav -o %s-normalized_2.wav'%(filename, basefile))
		# filenames.append('%s-normalized_2.wav'%(basefile))
		
		# using high volume 
		# os.system('ffmpeg-normalize %s high.wav -o file1-normalized.wav -o %s-normalized_3.wav'(filename, basefile))
		# filenames.append('%s-normalized_3.wav'%(basefile))
		
		# increase volume by 2x 
		new_file=change_volume(filename, 3)
		filenames.append(new_file)
		# decrease volume by 1/2 
		new_file=change_volume(filename, 0.33)
		filenames.append(new_file)

		return filenames 

	def normalize_pitch(filename):
		'''
		takes in an audio file and outputs files normalized to 
		different pitches. This corrects for gender ane time-of-day differences.

		where gives the pitch shift as positive or negative ‘cents’ (i.e. 100ths of a semitone). 
		There are 12 semitones to an octave, so that would mean ±1200 as a parameter.
		'''
		filenames=list()

		basefile=filename[0:-4]
		# down two octave 
		# os.system('sox %s %s pitch -2400'%(filename, basefile+'_freq_0.wav'))
		# filenames.append(basefile+'_freq_0.wav')

		# down two octave 
		os.system('sox %s %s pitch -600'%(filename, basefile+'_freq_one.wav'))
		filenames.append(basefile+'_freq_one.wav')

		# up one octave 
		os.system('sox %s %s pitch 600'%(filename, basefile+'_freq_two.wav'))
		filenames.append(basefile+'_freq_two.wav')

		# up two octaves 
		# os.system('sox %s %s pitch 2400'%(filename, basefile+'_freq_3.wav'))
		# filenames.append(basefile+'_freq_3.wav')

		return filenames 


	def time_stretch(filename):
		'''
		stretches files by 0.5x, 1.5x, and 2x.
		'''
		basefile=filename[0:-4]
		filenames=list()

		y, sr = librosa.load(filename)

		y_fast = librosa.effects.time_stretch(y, 1.5)
		librosa.output.write_wav(basefile+'_stretch_one.wav', y_fast, sr)
		filenames.append(basefile+'_stretch_one.wav')

		# y_fast_2 = librosa.effects.time_stretch(y, 1.5)
		# librosa.output.write_wav(basefile+'_stretch_1.wav', y, sr)
		# filenames.append(basefile+'_stretch_1.wav')

		y_slow = librosa.effects.time_stretch(y, 0.75)
		librosa.output.write_wav(basefile+'_stretch_two.wav', y_slow, sr)
		filenames.append(basefile+'_stretch_two.wav')

		# y_slow_2 = librosa.effects.time_stretch(y, 0.25)
		# librosa.output.write_wav(basefile+'_stretch_3.wav', y, sr)
		# filenames.append(basefile+'_stretch_3.wav')

		return filenames 

	def codec_enhance(filename, opusdir):

		filenames=list()
		#########################
		# lossy codec - .mp3
		#########################
		# os.system('ffmpeg -i %s %s'%(filename, filename[0:-4]+'.mp3'))
		# os.system('ffmpeg -i %s %s'%(filename[0:-4]+'.mp3', filename[0:-4]+'_mp3.wav'))
		# os.remove(filename[0:-4]+'.mp3')
		# filenames.append(filename[0:-4]+'_mp3.wav')

		#########################
		# lossy codec - .opus 
		#########################
		curdir=os.getcwd()
		newfile=filename[0:-4]+'.opus'

		# copy file to opus encoding folder 
		shutil.copy(curdir+'/'+filename, opusdir+'/'+filename)
		os.chdir(opusdir)
		print(os.getcwd())
		# encode with opus codec 
		os.system('opusenc %s %s'%(filename,newfile))
		os.remove(filename)
		filename=filename[0:-4]+'_opus.wav'
		os.system('opusdec %s %s'%(newfile, filename))
		os.remove(newfile)
		# delete .wav file in original dir 
		shutil.copy(opusdir+'/'+filename, curdir+'/'+filename)
		os.remove(filename)
		os.chdir(curdir)
		filenames.append(filename[0:-4]+'.wav')

		return filenames 


	def trim_silence(filename):
		new_filename=filename[0:-4]+'_trimmed.wav'
		command='sox %s %s silence -l 1 0.1 1'%(filename, new_filename)+"% -1 2.0 1%"
		os.system(command)

		return [new_filename]

	def remove_noise(filename):

		'''
		following remove_noise.py from voicebook.
		'''

		#now use sox to denoise using the noise profile
		data, samplerate =sf.read(filename)
		duration=data/samplerate
		first_data=samplerate/10
		filter_data=list()
		for i in range(int(first_data)):
		    filter_data.append(data[i])
		noisefile='noiseprof.wav'
		sf.write(noisefile, filter_data, samplerate)
		os.system('sox %s -n noiseprof noise.prof'%(noisefile))
		filename2='tempfile.wav'
		filename3='tempfile2.wav'
		noisereduction="sox %s %s noisered noise.prof 0.21 "%(filename,filename2)
		command=noisereduction
		#run command 
		os.system(command)
		print(command)
		#reduce silence again
		#os.system(silenceremove)
		#print(silenceremove)
		#rename and remove files 
		os.rename(filename2,filename[0:-4]+'_noise_remove.wav')
		#os.remove(filename2)
		os.remove(noisefile)
		os.remove('noise.prof')

		return [filename[0:-4]+'_noise_remove.wav']

	def add_noise(filename,curdir, newfilename):
		if filename[-4:]=='.wav':
			audioseg=AudioSegment.from_wav(filename)
		elif filename[-4:]=='.mp3':
			audioseg=AudioSegment.from_mp3(filename)
		hostdir=os.getcwd()
		os.chdir(curdir)
		os.chdir('noise')
		listdir=os.listdir()
		if 'noise.wav' in listdir:
			os.remove('noise.wav')
		mp3files=list()
		for i in range(len(listdir)):
			if listdir[i][-4:]=='.mp3':
				mp3files.append(listdir[i])
		noise=random.choice(mp3files)
		# add noise to the regular file 
		noise_seg = AudioSegment.from_mp3(noise)
		# find number of noise segments needed
		cuts=math.floor(len(audioseg)/len(noise_seg))
		noise_seg_2=noise_seg * cuts
		noise_seg_3=noise_seg[:(len(audioseg)-len(noise_seg_2))] 
		noise_seg_4=noise_seg_2 + noise_seg_3
		os.chdir(hostdir)
		print(len(noise_seg_4))
		print(len(audioseg))
		noise_seg_4.export("noise.wav", format="wav")
		# now combine audio file and noise file 
		os.system('ffmpeg -i %s -i %s -filter_complex "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]" -map "[a]" %s'%(filename, 'noise.wav',filename[0:-4]+'_noise.wav'))
		os.remove('noise.wav')
		os.rename(filename[0:-4]+'_noise.wav', newfilename)

		return [newfilename] 

	def random_splice(filename):
		# need to do this only in non-speaking regions. Need pause detection for this. will do later.
		return ''

	def insert_pauses(filename):
		# need to do this only in non-speaking regions. Need pause detection for this. will do later.
		return ''

	# _0=microphone_adjust(filename)
	_1=normalize_volume(filename)
	_2=normalize_pitch(filename)
	_3=time_stretch(filename)
	_4=codec_enhance(filename, opusdir)
	_5=trim_silence(filename)
	_6=remove_noise(filename)
	_7=add_noise(filename, curdir, filename[0:-4]+'_add_noise_one.wav')
	_8=add_noise(filename, curdir, filename[0:-4]+'_add_noise_two.wav')
	# _9=random_splice(filename)
	# _10=insert_pauses(filename)

	augmented_filenames=_1+_2+_3+_4+_5+_6+_7+_8

	return augmented_filenames

## augment by 'python3 augment.py [folderpath]
## '/Users/jim/desktop/files'
opusdir=os.getcwd()+'/opustools'
directory=sys.argv[1]
curdir=os.getcwd()
os.chdir(directory)
remove_json()
convert_audio()
time.sleep(5)
listdir=os.listdir()
wavfiles=list()

for i in range(len(listdir)):
	if listdir[i][-4:] in ['.wav']:
		new_name=listdir[i].replace(' ','_')
		os.rename(listdir[i],new_name)
		wavfiles.append(new_name)

print(wavfiles)

augmented_files=list()

for i in range(len(wavfiles)):
	try:
		print(os.getcwd())
		augmented_files=augment_dataset(wavfiles[i], opusdir, curdir)
		print(augmented_files)
		augsort(wavfiles[i], augmented_files, directory)
		print(augmented_files)
		
	except:
		print('error')

print('augmented dataset with %s files'%(str(len(augmented_files))))

