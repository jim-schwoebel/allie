import os, shutil
from tqdm import tqdm

def copy_files(directory_):
	listdir=os.listdir()
	newdir=os.getcwd()
	for i in tqdm(range(len(listdir)), desc=newdir):
		if listdir[i].endswith('.json') or listdir[i].endswith('.wav'):
			shutil.copy(os.getcwd()+'/'+listdir[i], directory_+'/'+listdir[i])

curdir=os.getcwd()
noise_dir=curdir+'/noise_combined'
voice_dir=curdir+'/voice_combined'

# os.chdir('noise_audioset')
# copy_files(noise_dir)

# os.chdir(curdir)
# os.chdir('noise_freespeech')
# copy_files(noise_dir)

# os.chdir(curdir)
# os.chdir('voice_audioset')
# copy_files(voice_dir)

# os.chdir(curdir)
# os.chdir('voice_freespeech')
# copy_files(voice_dir)

os.chdir(curdir)
os.chdir('noise_voxceleb')
copy_files(noise_dir)

os.chdir(curdir)
os.chdir('voice_voxceleb')
copy_files(voice_dir)


