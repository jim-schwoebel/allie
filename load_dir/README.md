# load_dir

Use this directory to make predictions on audio, text, image, video, and/or CSV files. 

Specifically, just drag and drop sample data in here and predictions will be made based on the models in the ./models directory.

## Getting started 

First, you need to put some files in the ./load_dir. I put the test files from the ./test directory into the load_dir here for demostration purposes. This is the output from the terminal (of files in the ./load_dir.

```
Jims-MBP:allie jimschwoebel$ cd load_dir
Jims-MBP:load_dir jimschwoebel$ ls
README.md	test_csv.csv	test_text.txt
test_audio.wav	test_image.png	test_video.mp4
```

Now you can apply all models in the directory according to sample types (audio_models, text_models, image_models, video_models, csv_models) by running the load_models.py command in this directory. 

```
cd ..
cd models 
python3 load_models.py
```

All the files will then be modeled.

## Supported file formats

Here are the supported file formats for the load directory. 

| File type | extension | recommended format | 
| ------------- |-------------| -------------| 
| audio file | .WAV, .MP3 | .WAV | 
| text file | .TXT | .TXT | 
| image file | .PNG, .JPG | .PNG | 
| video file | .MP4 | .MP4 | 
| CSV file | .CSV | .CSV | 
