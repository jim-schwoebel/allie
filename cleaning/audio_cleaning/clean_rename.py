import os, uuid

def clean_rename(audiofile):
	# replace wavfile with a version that is 16000 Hz mono audio
	os.rename(audiofile, str(uuid.uuid4())+audiofile[-4:])