'''
Taken from https://github.com/Shahabks/myprosody/blob/master/EXAMPLES.pdf
'''
import parselmouth
from parselmouth.praat import call, run_file
import numpy as np
import os, sys, shutil

def prev_dir(directory):
    g=directory.split('/')
    # print(g)
    lastdir=g[len(g)-1]
    i1=directory.find(lastdir)
    directory=directory[0:i1]
    return directory

def myprosody_featurize(wavfile):
	sound=os.getcwd()+'/'+wavfile
	path=os.getcwd()
	sourcerun=prev_dir(os.getcwd())+"helpers/myprosody/myprosody/"+"dataset"+"/"+"essen"+"/"+"myspsolution.praat"
	objects= run_file(sourcerun, -20, 2, 0.3, "yes",sound,path, 80, 400, 0.01, capture_output=True)
	print (objects[0]) # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
	z1=str( objects[1]) # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
	z2=z1.strip().split()
	z3=np.array(z2)
	z4=np.array(z3)[np.newaxis]
	z5=z4.T
	dataset={"number_ of_syllables":float(z5[0,:]),"number_of_pauses":float(z5[1,:]),"rate_of_speech":float(z5[2,:]),
			 "articulation_rate":float(z5[3,:]),"speaking_duration":float(z5[4,:]),"original_duration":float(z5[5,:]),
			 "balance":float(z5[6,:]),"f0_mean":float(z5[7,:]),"f0_std":float(z5[8,:]),"f0_median":float(z5[9,:]),
			 "f0_min":float(z5[10,:]),"f0_max":float(z5[11,:]),"f0_quantile25":float(z5[12,:]),"f0_quan75":float(z5[13,:])}


	# sound = parselmouth.Sound(wavfile)
	# formant = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=5500)
	# zero=formant.get_value_at_time(3, 0.5) # For the value of formant 3 at 0.5 seconds
	# print(zero)

	# other features you could extract 
	# one=sound.get_energy()
	# two=sound.get_energy_in_air()
	# three=sound.get_intensity()
	# five=sound.get_power()
	# six=sound.get_power_in_air()
	# seven=sound.get_rms()
	# eight=sound.get_root_mean_square()
	# nine=sound.to_harmonicity()
	# ten=sound.to_harmonicity_ac()
	# eleven=sound.to_harmonicity_cc()
	# twelve=sound.to_harmonicity_gne()
	# thirteen=sound.to_intensity()
	# fourteen=sound.to_mfcc()
	# fifteen=sound.to_pitch_ac()
	# sixteen=sound.to_pitch_cc()
	# seventeen=sound.to_pitch_shs()
	# eighteen=sound.to_pitch_spinet()
	# nineteen=sound.to_spectrogram()

	labels=list(dataset)
	features=list(dataset.values())

	return features, labels 

# os.chdir('test')
# wavfile='test.wav'
# cur_dir=os.getcwd()
# features, labels = myprosody_featurize(wavfile)
# print(features)
# print(labels)