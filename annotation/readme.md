## How to annotate

```python3
python3 annotate.py --directory {dirname} --sampletype {audio}
```

Converting to csv

```
python3 annotate2csv --directory {dirname}
```

### things to do
- classification and/or regression problem (if regression, float, if classification, integer value)
- add classnames (to Optparser)
- add ability to split up each file into more files in the folder {audio file / video file --> audio file 1, 2, n...} 
