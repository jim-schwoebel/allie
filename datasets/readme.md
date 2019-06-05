# Datasets

You can quickly download any of these datasets with the datasets.py script. This uses fuzzy search to figure out what dataset you are trying to find. An exhaustive list of all the audio, text, image, and video datasets are listed below.

Note you can search for more datasets using Google DataSet search @ https://toolbox.google.com/datasetsearch or Kaggle @ https://www.kaggle.com/datasets.

You can always create datasets with [mTurk](https://towardsdatascience.com/how-i-created-a-40-000-labeled-audio-dataset-in-4-hours-of-work-and-500-17ad9951b180) and/or [SurveyLex](https://surveylex.com) as well.

## Audio datasets 

Lots of audio datasets on here. https://towardsdatascience.com/a-data-lakes-worth-of-audio-datasets-b45b88cd4ad

### Open source 
* [AudioSet](https://research.google.com/audioset/) - A large-scale dataset of manually annotated audio events.
* [Common Voice](https://voice.mozilla.org/) - Common Voice is Mozilla's initiative to help teach machines how real people speak.
* [LJ speech](https://keithito.com/LJ-Speech-Dataset/) - This is a public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books. A transcription is provided for each clip. Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours.
* [CMU Wilderness](http://festvox.org/cmu_wilderness/) - (noncommercial) - not available but a great one. 
* [Urban Sound Dataset](https://urbansounddataset.weebly.com/) - two datasets and a taxonomy for urban sound research.
* [Noisy dataset](https://datashare.is.ed.ac.uk/handle/10283/2791)- Clean and noisy parallel speech database. The database was designed to train and test speech enhancement methods that operate at 48kHz. 
* [Spoken Commands dataset](https://github.com/JohannesBuchner/spoken-command-recognition) - A large database of free audio samples (10M words), a test bed for voice activity detection algorithms and for recognition of syllables (single-word commands).
* [Freesound dataset](https://www.kaggle.com/c/freesound-audio-tagging-2019/data) - many different sound events. https://annotator.freesound.org/ and https://annotator.freesound.org/fsd/explore/
* [Karoldvl-ESC](https://github.com/karoldvl/ESC-50) - The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.
* [Librispeech](https://www.openslr.org/12) - LibriSpeech is a corpus of approximately 1000 hours of 16Khz read English speech derived from read audiobooks from the LibriVox project.
* [TedLIUM](https://www.openslr.org/51/) - The TED-LIUM corpus was made from audio talks and their transcriptions available on the TED website (noncommercial).
* [VoxForge](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/) - VoxForge was set up to collect transcribed speech for use with Free and Open Source Speech Recognition Engines.
* [Tatoeba](https://tatoeba.org/eng/downloads) - Tatoeba is a large database of sentences, translations, and spoken audio for use in language learning. This download contains spoken English recorded by their community.
* [Speech accent archive](https://www.kaggle.com/rtatman/speech-accent-archive/version/1) - For various accent detection tasks.
* [The Emotional Voices Database](https://github.com/numediart/EmoV-DB) - various emotions with 5 voice actors (amused, angry, disgusted, neutral, sleepy).
* [TIMIT dataset](https://catalog.ldc.upenn.edu/LDC93S1) - Linguistic data consortium dataset for speech recogniition. Costs money and is not free, so may skip this one.

### NeuroLex datasets (need IRB access)
* [JamesVM dataset]() - 170,000+ voicemails left for loved ones around key events like anniversaries or birthdays. 1,000 of these files are annotated. 
* [MHA dataset]() - 350 self-reported patients with voice tasks and PHQ-9 depression labels.
* [YouTube disease dataset]() - >30 people in each category using YouTube videos (audio only).
* [Voiceome dataset]() - working on creating the world's largest dataset to tie voice information to health traits.
* [Framingham Heart Study dataset]() - 200 patients with transcriptions (manual) and neuropsychological testing for Alzheimer's and other areas. 
* [UW research dataset]() - some data associated labels from research assistants collecting data with disease labels (through REDCAP). 
* [Train-emotions]() - emotion labels using deep learning models + audio. 

## Text datasets
many listed https://machinelearningmastery.com/datasets-natural-language-processing/ and https://github.com/niderhoff/nlp-datasets
* [Sarcasm detection]() - Kaggle dataset off Redditt. 

## Image datasets
* [MNIST](http://yann.lecun.com/exdb/mnist/) - hand-written image dataset, standard to measure accuracy from Yan Lecun @ NYU.
* [MS-COCO](http://cocodataset.org/#download) - COCO is a large-scale object detection, segmentation, and captioning dataset. 330K images (>200K labeled), 1.5 million object instances, 80 object categories, 91 stuff categories, 5 captions per image, 250,000 people with keypoints.
* [ImageNet](http://www.image-net.org/) - ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. 
* [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) - 15,851,536 boxes on 600 categories, 2,785,498 instance segmentations on 350 categories, 36,464,560 image-level labels on 19,959 categories, 391,073 relationship annotations of 329 relationships, Extension - 478,000 crowdsourced images with 6,000+ categories.
* [VisualQA](https://visualqa.org/) - VQA is a new dataset containing open-ended questions about images. These questions require an understanding of vision, language and commonsense knowledge to answer. 265,016 images (COCO and abstract scenes)
At least 3 questions (5.4 questions on average) per image, 10 ground truth answers per question, 3 plausible (but likely incorrect) answers per question, Automatic evaluation metric.
* [The Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/) - 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10. 73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples, to use as extra training data.
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) - The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

## Video datasets
For a list of all the open source videos + papers below, check out: https://www.di.ens.fr/~miech/datasetviz/

### Open source 
* [VoxCeleb](https://github.com/andabi/voice-vector) - VoxCeleb videos.  1,251 Hollywood stars' 145,379 utterances
* [Lip reading dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/) - LRW, LRS2 and LRS3 are audio-visual speech recognition datasets collected from in the wild videos. 6M+ word instances, 800+ hours, 5,000+ identities
* [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) - HMDB: A Large Video Database for Human Motion Recognition - Action recognition	6766 videos,	51 action classes	pre-defined classes.
* [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) - UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild. 	13320	videos, 101 action classes	pre-defined classes
* [Sports-1M](http://cs.stanford.edu/people/karpathy/deepvideo/) - The YouTube Sports-1M Dataset. 1100000	videos, 487 sports classes pre-defined classes.
* [Charades](http://allenai.org/plato/charades/) - This dataset guides our research into unstructured video activity recognition and commonsense reasoning for daily human activities. 9848	 videos, 157 action labels, 27847 Free-text descriptions, action intervals, classes of interacted objects	pre-defined classes, text, intervals.
* [ActivityNet](http://activity-net.org/) - A Large-Scale Video Benchmark for Human Activity Understanding. 28000 videos,	203 classes	pre-defined classes.
* [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) - Kinetics is a large-scale, high-quality dataset of YouTube video URLs which include a diverse range of human focused actions. 500000	videos, 600 action classes	pre-defined classes.
* [YouTube 8M](https://research.google.com/youtube8m/download.html) - YouTube-8M is a large-scale labeled video dataset that consists of millions of YouTube video IDs and associated labels from a diverse vocabulary of 4700+ visual entities. 8000000 videos,	4716 classes	pre-defined classes.
* [AVA](https://research.google.com/ava/) - A Video Dataset of Spatio-temporally Localized Atomic Visual Actions. 57600 videos, 210k action labels, 80 atomic visual actions, spatio-temporal annotations	pre-defined classes, text, spatio-temporal annotation.
* [20BN-SOMETHING-SOMETHING](https://www.twentybn.com/datasets/something-something) - The 20BN-SOMETHING-SOMETHING dataset is a large collection of densly-labeled video clips that show humans performing predefined basic actions with every day objects. 108000 videos, 174 classes	pre-defined classes.
* [20BN-JESTER](https://www.twentybn.com/datasets/jester) - Human Hand Gestures Dataset. 148000 videos	27 classes	pre-defined classes.
* [LSMDC](http://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/mpii-movie-description-dataset/) - Large-Scale Movie Understanding Dataset. 118000 videos, Aligned captions	text.
* [DALY](http://thoth.inrialpes.fr/daly/) - Daily Action Localization in Youtube videos. 8100 videos, 3.6k spatio-temporal action annotation	pre-defined classes, spatio-temporal annotation
* [MPII-Cooking](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/human-activity-recognition/mpii-cooking-2-dataset/) - MPII Cooking dataset. 273	videos, 78 classes, 13k labelled instances	pre-defined classes, text.
* [Hollywood2](http://www.di.ens.fr/~laptev/actions/hollywood2/) - Human actions and scenes dataset. 3669 videos, 12 human action classes, 10 classes of scene	pre-defined classes
* [VideoMCC](http://videomcc.org/) - 272000	videos, 10 topics and Video captions	Question-Answer, text.
* [MovieQA](http://movieqa.cs.toronto.edu/home/) - Movie Understanding	140	15k Question-Answer, 408 movie plots, 408 subtitles	Question-Answer, text.
* [ActivityNet Captions](http://cs.stanford.edu/people/ranjaykrishna/densevid/) - a large-scale benchmark for dense-captioning events.	20000	videos, 100k Aligned captions	text.
* [Youtube BoundingBoxes](https://research.google.com/youtube-bb/) - YouTube-BoundingBoxes Dataset. 240000 videos 5.6M Bouding boxes, 23 objects	Bounding boxes.
* [DAVIS](http://davischallenge.org/) - Densely Annotated VIdeo Segmentation. 50	videos 3455 annotated frames	Segmentation mask.
* [FCVID](http://bigvid.fudan.edu.cn/FCVID/) - Fudan-Columbia Video Dataset - 91223 videos 239 classes	pre-defined classes
* [VGG Human Pose](https://www.robots.ox.ac.uk/~vgg/data/pose/index.html) - The VGG Human Pose Estimation datasets is a set of large video datasets annotated with human upper-body pose. 152	videos Hours of human upper-body pose	human pose.
* [YFCC100M](http://yfcc100m.appspot.com/?) - YFCC100M: The New Data in Multimedia Research. 800000	videos, 1570 tags, captions and diverse metadata	Captions, pre-defined classes.
* [ASLAN](http://www.openu.ac.il/home/hassner/data/ASLAN/ASLAN.html) - ASLAN. The Action Similarity Labeling dataset. 1571	videos, 432 action classes, 3697 action samples	pre-defined classes
* [Instruction Video Dataset](http://www.di.ens.fr/willow/research/instructionvideos/) - A new challenging dataset of real-world instruction videos from the Internet. 150	videos, 5 different instructional tasks with subtitles	pre-defined classes, captions
* [DiDeMo dataset](https://people.eecs.berkeley.edu/~lisa_anne/didemo.html) - the Distinct Describable Moments (DiDeMo) dataset consists of over 10,000 unedited, personal videos in diverse visual settings with pairs of localized video segments and referring expressions. 10000 videos,	40000 aligned captions	captions
* [MSR-VTT](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/) - A Large Video Description Dataset for Bridging Video and Language. 10000	videos, 200000 aligned captions	captions.
* [SLAC](http://slac.csail.mit.edu/) - A Sparsely Labeled ACtions Dataset. 520000	videos, 200 action classes, 1.75M clip annotations	pre-defined classes.
* [VLOG](https://people.eecs.berkeley.edu/~dfouhey/2017/VLOG/index.html) - VLOG From Lifestyle VLOGs to Everyday Interactions: The VLOG Dataset. 114000 videos	pre-defined classes.
* [Moments in Time](http://moments.csail.mit.edu/) - Moments in Time Dataset: one million videos for event understanding. 1000000	videos, 339 action classes	pre-defined classes
* [TGIF](http://raingo.github.io/TGIF-Release/) - TUMBLR GIF captioning	125781	videos, 125781 captions	

### NeuroLex datasets
* [YouTube disease dataset]() - >30 people in each category using YouTube videos (videos). 

## CSV datasets 
### Open source
* see [PyDataset](https://github.com/iamaziz/PyDataset)

### NeuroLex datasets
* TRIBE 4 application.
