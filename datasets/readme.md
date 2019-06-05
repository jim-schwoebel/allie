# Datasets

You can quickly download any of these datasets with the datasets.py script. This uses fuzzy search to figure out what dataset you are trying to find. An exhaustive list of all the audio, text, image, and video datasets are listed below.

Note you can search for more datasets using Google DataSet search @ https://toolbox.google.com/datasetsearch or Kaggle @ https://www.kaggle.com/datasets.

You can always create datasets with [mTurk](https://towardsdatascience.com/how-i-created-a-40-000-labeled-audio-dataset-in-4-hours-of-work-and-500-17ad9951b180) and/or [SurveyLex](https://surveylex.com) as well.

## Audio datasets 
There are two main types of audio datasets: speech datasets and audio event/music datasets.

### Speech datasets 
* [Common Voice](https://voice.mozilla.org/) - Common Voice is Mozilla's initiative to help teach machines how real people speak. 12GB in size; spoken text based on text from a number of public domain sources like user-submitted blog posts, old books, movies, and other public speech corpora.
* [LJ speech](https://keithito.com/LJ-Speech-Dataset/) - This is a public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books. A transcription is provided for each clip. Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours.
* [CMU Wilderness](http://festvox.org/cmu_wilderness/) - (noncommercial) - not available but a great speech dataset many accents reciting passages from the Bible.
* [Noisy dataset](https://datashare.is.ed.ac.uk/handle/10283/2791)- Clean and noisy parallel speech database. The database was designed to train and test speech enhancement methods that operate at 48kHz. 
* [Spoken Commands dataset](https://github.com/JohannesBuchner/spoken-command-recognition) - A large database of free audio samples (10M words), a test bed for voice activity detection algorithms and for recognition of syllables (single-word commands). 3 speakers, 1,500 recordings (50 of each digit per speaker), English pronunciations. This is a really small set- about 10 MB in size.
* [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) -4 speakers, 2,000 recordings (50 of each digit per speaker), English pronunciations.
* [Speech Commands Dataset](http://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) - The dataset (1.4 GB) has 65,000 one-second long utterances of 30 short words, by thousands of different people, contributed by members of the public through the AIY website.
* [Librispeech](https://www.openslr.org/12) - LibriSpeech is a corpus of approximately 1000 hours of 16Khz read English speech derived from read audiobooks from the LibriVox project.
* [Ted-LIUM](https://www.openslr.org/51/) - The TED-LIUM corpus was made from audio talks and their transcriptions available on the TED website (noncommercial).
* [VoxForge](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/) - VoxForge was set up to collect transcribed speech for use with Free and Open Source Speech Recognition Engines.
* [VoxCeleb](https://github.com/andabi/voice-vector) - VoxCeleb is a large-scale speaker identification dataset. It contains around 100,000 utterances by 1,251 celebrities, extracted from You Tube videos. The data is mostly gender balanced (males comprise of 55%). The celebrities span a diverse range of accents, professions, and age. There is no overlap between the development and test sets. It’s an intriguing use case for isolating and identifying which superstar the voice belongs to.
* [Tatoeba](https://tatoeba.org/eng/downloads) - Tatoeba is a large database of sentences, translations, and spoken audio for use in language learning. This download contains spoken English recorded by their community.
* [Speech accent archive](https://www.kaggle.com/rtatman/speech-accent-archive/version/1) - For various accent detection tasks.
* [The Emotional Voices Database](https://github.com/numediart/EmoV-DB) - various emotions with 5 voice actors (amused, angry, disgusted, neutral, sleepy).
* [TIMIT dataset](https://catalog.ldc.upenn.edu/LDC93S1) - TIMIT contains broadband recordings of 630 speakers of eight major dialects of American English, each reading ten phonetically rich sentences. It includes time-aligned orthographic, phonetic and word transcriptions as well as a 16-bit, 16 kHz speech waveform file for each utterance (have to pay).
* [Spoken Wikipeida Corpora](https://nats.gitlab.io/swc/) - 38 GB in size available in both audio and without audio format.
* [Flickr Audio Caption](https://groups.csail.mit.edu/sls/downloads/flickraudio/) - 40,000 spoken captions of 8,000 natural images, 4.2 GB in size.
* [Persian Consonant Vowel Combination (PCVC) Speech Dataset](https://github.com/S-Malek/PCVC) - The Persian Consonant Vowel Combination (PCVC) Speech Dataset is a Modern Persian speech corpus for speech recognition and also speaker recognition. This dataset contains 23 Persian consonants and 6 vowels. The sound samples are all possible combinations of vowels and consonants (138 samples for each speaker) with a length of 30000 data samples.
* [CHIME](https://archive.org/details/chime-home) - This is a noisy speech recognition challenge dataset (~4GB in size). The dataset contains real simulated and clean voice recordings. Real being actual recordings of 4 speakers in nearly 9000 recordings over 4 noisy locations, simulated is generated by combining multiple environments over speech utterances and clean being non-noisy recordings. 
* [2000 HUB5 English](https://catalog.ldc.upenn.edu/LDC2002T43) - The Hub5 evaluation series focused on conversational speech over the telephone with the particular task of transcribing conversational speech into text. Its goals were to explore promising new areas in the recognition of conversational speech, to develop advanced technology incorporating those ideas and to measure the performance of new technology.
* [Parkinson's speech dataset](https://archive.ics.uci.edu/ml/datasets/Parkinson+Speech+Dataset+with++Multiple+Types+of+Sound+Recordings) - The training data belongs to 20 Parkinson’s Disease (PD) patients and 20 healthy subjects. From all subjects, multiple types of sound recordings (26) are taken for this 20 MB set.
* [Zero Resource Speech Challenge](https://github.com/bootphon/zerospeech2017) - The ultimate goal of the Zero Resource Speech Challenge is to construct a system that learns an end-to-end Spoken Dialog (SD) system, in an unknown language, from scratch, using only information available to a language learning infant. “Zero resource” refers to zero linguistic expertise (e.g., orthographic/linguistic transcriptions), not zero information besides audio (visual, limited human feedback, etc). The fact that 4-year-olds spontaneously learn a language without supervision from language experts show that this goal is theoretically reachable.
* [ISOLET Data Set](https://data.world/uci/isolet) - This 38.7 GB dataset helps predict which letter-name was spoken — a simple classification task.
* [Arabic Speech Corpus](http://en.arabicspeechcorpus.com/) - The Arabic Speech Corpus (1.5 GB) is a Modern Standard Arabic (MSA) speech corpus for speech synthesis. The corpus contains phonetic and orthographic transcriptions of more than 3.7 hours of MSA speech aligned with recorded speech on the phoneme level. The annotations include word stress marks on the individual phonemes. 
* [Multimodal EmotionLines Dataset (MELD)](https://github.com/SenticNet/MELD) - Multimodal EmotionLines Dataset (MELD) has been created by enhancing and extending EmotionLines dataset. MELD contains the same dialogue instances available in EmotionLines, but it also encompasses audio and visual modality along with text. MELD has more than 1400 dialogues and 13000 utterances from Friends TV series. Each utterance in a dialogue has been labeled with— Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear. 

### Audio events and music 
* [AudioSet](https://research.google.com/audioset/) - An expanding ontology of 632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos. 
* [Bird audio detection challenge](http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/) -  This challenge contained new datasets (5.4 GB) collected in real live bio-acoustics monitoring projects, and an objective, standardized evaluation framework.
* [Environmental audio dataset](http://www.cs.tut.fi/~heittolt/datasets) - Audio data collection and manual data annotation both are tedious processes, and lack of proper development dataset limits fast development in the environmental audio research.
* [Free Music Archive](https://github.com/mdeff/fma) - FMA is a dataset for music analysis. 1000 GB in size.
* [Freesound dataset](https://www.kaggle.com/c/freesound-audio-tagging-2019/data) - many different sound events. https://annotator.freesound.org/ and https://annotator.freesound.org/fsd/explore/ - The AudioSet Ontology is a hierarchical collection of over 600 sound classes and we have filled them with 297,159 audio samples from Freesound. This process generated 678,511 candidate annotations that express the potential presence of sound sources in audio clips.
* [Karoldvl-ESC](https://github.com/karoldvl/ESC-50) - The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.
* [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/) - The Million Song Dataset is a freely-available collection of audio features and meta-data for a million contemporary popular music tracks. 280 GB in size.
* [Urban Sound Dataset](https://urbansounddataset.weebly.com/) - two datasets and a taxonomy for urban sound research.

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

## CSV datasets 
### Open source
* see [PyDataset](https://github.com/iamaziz/PyDataset)

## References
* [Audio datasets](https://towardsdatascience.com/a-data-lakes-worth-of-audio-datasets-b45b88cd4ad) 
* [Datasets-natural-language-processing](https://machinelearningmastery.com/datasets-natural-language-processing/) 
* [Google Dataset Search](https://toolbox.google.com/datasetsearch)
* [Kaggle](https://kaggle.com)
* [NP Datasets](https://github.com/niderhoff/nlp-datasets)
* [PyDataset](https://github.com/iamaziz/PyDataset)
* [Video datasets](https://www.di.ens.fr/~miech/datasetviz/)


