# installing_opensmile
installing opensmile for macs

Download xcode (mac) - https://gist.github.com/gaquino/87bdf0e6e852e445c0489379d3e9732a

Then run the following:
```
brew install autoconf automake m4 libtool
cd opensmile-2.3.0
bash autogen.sh
bash autogen.sh
./configure
make -j4 ; make
make install
```
# if linux: https://github.com/naxingyu/opensmile/issues/2
Now you can extract features:
```
SMILExtract -C config/emobase.conf -I data/0.wav -O data/output.arff
```
## various feature extractors

For use in the Allie framework:

- SMILExtract -C config/emobase.conf (989 features)
- SMILExtract -C config/emo_large.conf (6553 features)
...

```python3
feature_extractors=['avec2013.conf', 'emobase2010.conf', 'IS10_paraling.conf', 'IS13_ComParE.conf',     'IS10_paraling_compat.conf', 'emobase.conf', 'emo_large.conf', 'IS11_speaker_state.conf', 'IS12_speaker_trait_compat.conf', 'IS09_emotion.conf', 'IS12_speaker_trait.conf', 'prosodyShsViterbiLoudness.conf', 'ComParE_2016.conf', 'GeMAPSv01a.conf']
```

## reading various arff files

```python3 
def parseArff(self,arff_file):
    '''
    Parses Arff File created by OpenSmile Feature Extraction
    '''
    f = open(arff_file,'r')
    data = []
    labels = []
    for line in f:
        if '@attribute' in line:
            temp = line.split(" ")
            feature = temp[1]
            labels.append(feature)
        if ',' in line:
            temp = line.split(",")
            for item in temp:
                data.append(item)
    temp = arff_file.split('/')
    temp = temp[-1]
    data[0] = temp[:-5] + '.wav'
    return data,labels
```
