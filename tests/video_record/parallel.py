import os

# if just want one:
# os.system('python3 record.py test.avi 30')

# record 30 - can also make this contextual based on date/time
for i in range(30):
	os.system("python3 record.py %s 10"%(str(i)+'.avi'))
