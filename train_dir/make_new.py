import pandas as pd

data=pd.read_csv('new2.csv')
urls=data['url']
newurls=list()
for i in range(len(urls)):
	newurls.append('/mnt/c/users/jimsc/desktop/voiceome_analysis/allie/train_dir/audio-features-6535df16-e0f2-11ea-9425-380025122270/'+urls[i].split('/')[-1])
data['url']=newurls
data.to_csv('new3.csv', index=False)