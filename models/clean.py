import os, shutil

listdir=os.listdir()

for i in range(len(listdir)):
	if listdir[i] not in ['audio_models', 'readme.md'] and listdir[i].endswith('.py') == False:

		if listdir[i].find('.') == -1:
			shutil.rmtree(listdir[i])
		else:
			os.remove(listdir[i])

		print('removed %s'%(listdir[i]))
