# pyAudioLex
This is the main NeuroLex python library.

# Dependencies
```
pip install --no-cache-dir -r requirements.txt
python -m nltk.downloader -d /usr/share/nltk_data punkt averaged_perceptron_tagger universal_tagset
```

# Explanation of Features

## Linguistic features 
A list of the linguistic features that are extracted from the transcriptions of the recorded conversations with the test subjects is shown in Table 1. The features are geared towards detecting problems with the flow of the conversation and measuring how well the subject can understand the question or carry on the conversation without getting confused. Recordings are manually transcribed. Each recording is first split into conversation turns. Then, the turns where the subjects speak are further split into utterances that are segments where the subjects talk without interruption by the interviewer and without long silences. The splitting mechanism is shown in Figure 1 where a voice activity detector is used for detecting silences. Only the turns of subjects are used in feature extraction.

### Hesitation and puzzlement features
During recording, we found that patients tend to hesitate more, forget what they were talking about, and have a harder time finding the right words or remembering details about their pasts. They also sometimes get confused about why they cannot remember the details or forget the context of the conversation. Those observations led us to propose features that will be able to capture those patterns in transcriptions.

#### 1.1 Question ratio
Patients are more likely to forget details in the middle of conversation, to not understand the questions, or to forget the context of the question. In those cases, they tend to ask the interviewer to repeat the question or they get confused, talk to themselves, and ask further questions about the details. The question words such as 'which,' 'what,' etc. are tagged automatically in each conversation. The full list of question tags that were used here is shown in Table 2. The question ratio of a subject is computed by dividing the total number of question words by the number of utterances spoken by the subject.

```python
from nltk.tokenize import RegexpTokenizer, word_tokenize

tokenizer = RegexpTokenizer('Who|What|When|Where|Why|How|\?')

s = "What time is it? What day is it?"
qtokens = tokenizer.tokenize(s)
tokens = word_tokenize(s)

ratio = float(len(qtokens)) / float(len(tokens))
```

#### 1.2 Filler ratio
Filler sounds such as 'ahm' and 'ehm' are used by people in spoken language when they think about what to say next. We hypothesize that they may be used more frequently by the patients because of slow thinking and memory recall processes. Patients tend to forget what they are talking about and to use fillers more often than the control subjects. The filler ratio is computed by dividing the total number of filler words by the total number of utterances spoken by the subject.

```python
from nltk.tokenize import RegexpTokenizer, word_tokenize

tokenizer = RegexpTokenizer('uh|ugh|um|like|you know')

s = "Uh I am, like, hello, you know, and um or umm."
qtokens = tokenizer.tokenize(s.lower())
tokens = word_tokenize(s)

ratio = float(len(qtokens)) / float(len(tokens))
```

#### 1.3 Incomplete sentence ratio
One of our observations of the patients is their inability to complete sentences. They seem to either forget what they were going to say or to completely change the context and start talking about a different topic. Incomplete sentences are manually labeled for each conversation. To compute this feature, the ratio of incomplete sentences to the total number of the sentences is calculated.

```python
# undetermined at this point
```

### POS-based features 
Part of speech (POS) tags can be used to extract markers for detecting the disease. For example, frequent adjectives can indicate more colorful and descriptive use of language, while frequent adverbs can indicate the ability to relate different utterances to each other. The frequency of each POS tag can also be a useful identifier of patients with Alzheimer's disease.

POS tags are added automatically to each word using a Turkish stemmer [18]. In cases where a word can have multiple alternative POS tags, equal weights are given to all possibilities. For instance, if a word can be either a noun or an adverb, depending on the sentence, that word is counted as half adverb and half noun in computation.

The following POS tag frequencies are used as features:

- Verb frequency
- Noun frequency
- Pronoun frequency
- Adverb frequency
- Adjective frequency
- Particle frequency
- Conjunction frequency
- Pronoun-to-noun ratio

Frequency of a POS tag is computed by dividing the total number of words with that tag by the total number of words spoken by the subject in the recording. Pronoun-to-noun ratio is the ratio of the total number of pronouns to the total number of nouns.

#### 2.1 Verb freq.

```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag

s = "They refuse to permit us to obtain the refuse permit"
tokens = word_tokenize(s)
pos = pos_tag(tokens)

verbs = []
for [token, tag] in pos:
  part = map_tag('en-ptb', 'universal', tag)
  if part == "VERB":
    verbs.append(token)

frequency = float(len(verbs)) / float(len(tokens))
```

#### 2.2 Noun freq.

```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag

s = "They refuse to permit us to obtain the refuse permit"
tokens = word_tokenize(s)
pos = pos_tag(tokens)

nouns = []
for [token, tag] in pos:
  part = map_tag('en-ptb', 'universal', tag)
  if part == "NOUN":
    nouns.append(token)

frequency = float(len(nouns)) / float(len(tokens))
```

#### 2.3 Pronoun freq.

```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag

s = "They refuse to permit us to obtain the refuse permit"
tokens = word_tokenize(s)
pos = pos_tag(tokens)

pronouns = []
for [token, tag] in pos:
  part = map_tag('en-ptb', 'universal', tag)
  if part == "PRON":
    pronouns.append(token)

frequency = float(len(pronouns)) / float(len(tokens))
```

#### 2.4 Adverb freq.

```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag

s = "They refuse to permit us to obtain the refuse permit"
tokens = word_tokenize(s)
pos = pos_tag(tokens)

adverbs = []
for [token, tag] in pos:
  part = map_tag('en-ptb', 'universal', tag)
  if part == "ADV":
    adverbs.append(token)

frequency = float(len(adverbs)) / float(len(tokens))
```

#### 2.5 Adjective freq.

```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag

s = "They refuse to permit us to obtain the refuse permit"
tokens = word_tokenize(s)
pos = pos_tag(tokens)

adjectives = []
for [token, tag] in pos:
  part = map_tag('en-ptb', 'universal', tag)
  if part == "ADJ":
    adjectives.append(token)

frequency = float(len(adjectives)) / float(len(tokens))
```

#### 2.6 Particle freq.

```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag

s = "They refuse to permit us to obtain the refuse permit"
tokens = word_tokenize(s)
pos = pos_tag(tokens)

particles = []
for [token, tag] in pos:
  part = map_tag('en-ptb', 'universal', tag)
  if part == "PRT":
    particles.append(token)

frequency = float(len(particles)) / float(len(tokens))
```

#### 2.7 Conjunction freq.

```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag

s = "They refuse to permit us to obtain the refuse permit"
tokens = word_tokenize(s)
pos = pos_tag(tokens)

conjunctions = []
for [token, tag] in pos:
  part = map_tag('en-ptb', 'universal', tag)
  if part == "CONJ":
    conjunctions.append(token)

frequency = float(len(conjunctions)) / float(len(tokens))
```

#### 2.8 Pronoun-to-noun ratio
Pronoun-to-noun ratio is the ratio of the total number of pronouns to the total number of nouns.

```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag

s = "They refuse to permit us to obtain the refuse permit"
tokens = word_tokenize(s)
pos = pos_tag(tokens)

pronouns = []
for [token, tag] in pos:
  part = map_tag('en-ptb', 'universal', tag)
  if part == "PRON":
    pronouns.append(token)

nouns = []
for [token, tag] in pos:
  part = map_tag('en-ptb', 'universal', tag)
  if part == "NOUN":
    nouns.append(token)

frequency = float(len(pronouns)) / float(len(nouns))
```

### Intelligibility Complexity features

#### 3.0 Unintelligible word ratio
During the conversations, some of the words spoken by the patients were unintelligible. These are mostly because patients could not produce the words correctly, they mumbled, or they were thinking while talking, which reduced intelligibility. Unintelligible word ratio is the ratio of unintelligible words to all words spoken by the subject.

Annotation of unintelligible words was done manually by three listeners for each conversation. A word was marked as unintelligible only when at least two of the three listeners could not understand it.

```python
# undetermined at this point
```

#### 4.1 Standardized word entropy
One of the earliest parts of the brain to be damaged by Alzheimer's disease is the part of the brain that deals with language ability [5]. We hypothesize that this may cause a degradation in the variety of words and word combinations that a patient uses. Standardized word entropy, i.e., word entropy divided by the log of the total word count, is used to model this phenomenon. Because the aim is to compute the variety of word choice, stemming is done, and only the stems of the words are considered.

```python
import math
from nltk import FreqDist
from nltk.tokenize import word_tokenize

def entropy(tokens):
  freqdist = FreqDist(tokens)
  probs = [freqdist.freq(l) for l in freqdist]
  return -sum(p * math.log(p,2) for p in probs)

s = "male female male female"
tokens = word_tokenize(s)
standardized_entropy = entropy(tokens) / math.log(len(tokens))
```

#### 4.2 Suffix ratio
The standardized word entropy feature focuses on the variety of the stem words while ignoring the suffixes. However, suffixes can also be strong indicators of the complexity of a sentence. Turkish, in particular, has a rich and complex morphological structure [19]. Hundreds of different words can be generated from the same stem word by appending suffixes to it. Thus, we investigated whether the patients tend to construct simpler words than the control subjects by analyzing the suffixes they used. The suffix ratio of a subject is calculated by dividing the total number of suffixes by the total number of words spoken by the subject.

```python
# undetermined at this point
```

#### 4.3 Number ratio
During conversations, subjects give details about their birth dates, how many kids they have, and other numerical information. Such use of numbers in a sentence can be a measure of recall ability. The number ratio feature is calculated by dividing the total count of numbers by the total count of words the subject used in the conversation.

```python
from nltk.tokenize import RegexpTokenizer, word_tokenize

tokenizer = RegexpTokenizer('zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|dozen|couple|several|few|\d')

s = "I found seven apples by a couple of trees. I found a dozen eggs by those 5 chickens."
qtokens = tokenizer.tokenize(s.lower())
tokens = word_tokenize(s)

ratio = float(len(qtokens)) / float(len(tokens))
```

#### 4.4 Brunet's index
Brunet's index (W) quantifies lexical richness [20]. It is calculated as W = N<sup>V<sup>−0.165</sup></sup>, where N is the total text length and V is the total vocabulary. Lower values of W correspond to richer texts. As with standardized word entropy, stemming is done on words and only the stems are considered.

```python
# Additionally, see: Pg. 50 http://www.csb.uncw.edu/mscsis/complete/pdf/Habash%20-%20Capstone%20Paper.pdf
import math
from nltk.tokenize import word_tokenize

s = "Bravely bold Sir Robin, rode forth from Camelot."
tokens = word_tokenize(s)

N = float(len(tokens))
V = float(len(set(tokens)))
W = math.pow(N, math.pow(V, -0.165))
```

#### 4.5 Honore's statistic
Honore's statistic [21] is based on the notion that the larger the number of words used by a speaker that occur only once, the richer his overall lexicon is. Words spoken only once (V1) and the total vocabulary used (V) have been shown to be linearly associated. Honore's statistic generates a lexical richness measure according to R = (100 × log(N)) / (1 − (V1 / V)), where N is the total text length. Higher values correspond to a richer vocabulary. As with standardized word entropy, stemming is done on words and only the stems are considered.

```python
# Additionally, see: Pg. 50 http://www.csb.uncw.edu/mscsis/complete/pdf/Habash%20-%20Capstone%20Paper.pdf
import math
from nltk.tokenize import word_tokenize
from nltk import FreqDist

s = "Bravely bold Sir Robin, rode forth from Camelot. Afterwards, Sir Robin went to the castle."
tokens = word_tokenize(s)
uniques = []

for token, count in FreqDist(tokens).items():
  if count == 1:
    uniques.append(token)

N = float(len(tokens))
V = float(len(set(tokens)))
V1 = float(len(uniques))
R = (100 * math.log(N)) / (1 - (V1 / V))
```

#### 4.6 Type-token ratio
A pattern that we noticed in the recordings of the Alzheimer's patients is the frequency of repetitions in conversation. Patients tend to forget what they have said and to repeat it elsewhere in the conversation. The metric that we used to measure this phenomenon is type-token ratio [22]. Type-token ratio is defined as the ratio of the number of unique words to the total number of words. In order to better assess the repetitions, only the stems of the words are considered in calculations.

```python
from nltk.tokenize import word_tokenize
from nltk import FreqDist

s = "Bravely bold Sir Robin, rode forth from Camelot. Afterwards, Sir Robin went to the castle."
tokens = word_tokenize(s)
uniques = []

for token, count in FreqDist(tokens).items():
  if count == 1:
    uniques.append(token)

ratio = float(len(uniques)) / float(len(tokens))
```

## Prosodic features 
A total of 20 prosodic features were extracted and evaluated for detecting Alzheimer's disease. A list of all prosodic features used here is shown in Table 1. Descriptions of the prosodic features are given below. All prosodic feature computations are performed over the locution of the subject. Locution is the total response period of the subject which is the sum of all of the subject's speech turns. Each speech turn includes utterances, long silences, and short silences, as shown in Figure 1.

### Voice activity-related features
Silence and speech segments are automatically labeled in each conversation with a voice activity detector (VAD). The VAD used here is based on the distribution of the short-time frame energy of the speech signal. Because there is both silence and speech in the recordings, the energy distribution has two modes, both of which can be modeled with a Gaussian distribution. The bimodal distribution of silence and speech is trained using the expectation-maximization (EM) algorithm. The mode that has a lower mean is used to represent silence, and the mode that has a higher mean is used to represent speech.

Energy of each short-time speech frame in the recording is classified as either speech or silence using the likelihood ratio test (LRT). Because the test treats each frame independently, a second processing step is used where silence and speech segments that were shorter than four frames are removed.

The transcriptions of recordings were available and could be used for VAD through forced alignment using an automatic speech recognition system. However, the VAD described above worked well andmore sophisticated VAD techniques were not required.

#### 5.1 Response time
When the interviewer asks a question, it often takes some time before the subject gives a response. It is hypothesized that this time can be an indicator of the disease since it is expected to be related to cognitive processes such as attention and memory. The time it takes the subject to answer a question is calculated in each segment as the response time measure.

```python
# undetermined at this point
```

#### 5.2 Response length
Response length is the average length of a subject's response in seconds to the interviewer's question. Beginning and trailing silences are removed.

```python
# undetermined at this point
```

#### 5.3 Silence ratio
The plan-and-execute cycle in speech production was found to be distinctly different in patients compared to control subjects as noted in [14]. In our data, we also observed that patients tend to stop more in the middle of sentences to think about what to say next. The silence ratio is computed by dividing the total number of silences in the whole locution by the total number of words in the locution. Dividing by the number of words, we reduce the variability that arises from different speaking rates.

```python
# undetermined at this point
```

#### 5.4 Silence to utt. ratio
Silence-to-utterance ratio is the ratio of the total number of silences to the total number of utterances. Similar to silence ratio, it is a measure of the hesitation rate of the subject.

```python
# undetermined at this point
```

#### 5.5 Long silence ratio
Patients sometimes pause for a long time while answering a question. They do not use fillers during these long periods, and the interviewers did not interrupt these periods of silence. We hypothesized that these pauses may correspond to moments when the subject is retrieving information which is expected to be longer for the Alzheimer's patients. Similarly, confusion may also lead to long silences. The rate of such long hesitation events, defined as silences longer than approximately one second, is used to detect the disease. This feature is computed as the ratio of the total count of long silences to the total number of words.

```python
# undetermined at this point
```

#### 5.6 Avg. silence count
This feature specifies the average number of silences produced by a speaker in one second of speech. It is calculated by dividing the total number of silences by the duration of the locution.

```python
# undetermined at this point
```

#### 5.7 Silence rate
The silence rate measures the silence as a proportion of the whole locution. It is computed by dividing the total duration of all silence segments by the duration of the locution.

```python
# undetermined at this point
```

#### 5.8 Cont. speech rate
This feature measures how long the subject speaks until the next long silence, which is considered to be a thinking or recalling state. It is defined as the average duration of continuous speech segments over the whole locution.

```python
# undetermined at this point
```

#### 5.9 Avg. cont. word count
As mentioned above, the thinking process longer for patients than for the control subjects. The silence rate features discussed above try to exploit this long thinking process. Another way tomeasure it is to compute the average number of consecutive words that are spoken without intervening silences. First, the number of words for each continuous segment is computed. Then, the mean of these counts is used as the feature.

```python
# undetermined at this point
```

### Articulation-related features
The voice activity-related features discussed above are related to cognitive thought processes. However, it is also important to measure how the subject uses his or her voice articulations during speech. For example, if the subject becomes emotional, significant changes in the fundamental frequency (pitch) can be expected. Similarly, changes in the resonant frequencies (formants) of speech can be a strong indicator of the subject's health. If the formants do not change fast enough or are not distinct enough, sounds may become harder for listeners to identify, leading to the perception of mumbling. In order to see the impact of these effects on classification of the disease, pitch and formant trajectories are extracted, and the following features are derived over the whole locution.

#### 6.1 Avg. abs. delta energy
Energy variations can convey information about the mood of the subject. Changing energy significantly during speech may indicate a conscious effort to stress words that are semantically important or a change in mood related to the content of the speech. The absolute value of each frame-to-frame energy change is measured, and the average of these changes over the whole locution is computed.

```python
# undetermined at this point
```

#### 6.2 Dev. of abs. delta energy
In addition to changes in energy, changes in the delta energy, which is the acceleration of energy, can be used. The standard deviation of the average absolute delta energy is used to further investigate the possible impacts of the disease on the energy change rate.

```python
# undetermined at this point
```

#### 6.3 Avg. abs. delta pitch
The average of the absolute delta pitch shows the rate of variations in pitch. This feature is highly correlated with the emotions carried through the speech signal.

```python
# undetermined at this point
```

#### 6.4 Dev. of abs. delta pitch
The standard deviation of the absolute delta pitch is also used as a feature to further analyze the possible impacts of the disease on the pitch change rate. A monotonic increase or decrease in the pitch may simply be related to routine changes in sentence structure. However, acceleration of pitch, measured with the standard deviation of absolute delta pitch, can capture unusual pitch events in speech.

```python
# undetermined at this point
```

#### 6.5.1 Avg. abs. delta formant 1
The average of the absolute delta formant frequencies indicates the rate of change in the formant features. Formants are related to the positions of the vocal organs such as the tongue and lips. Reduction of control over these organs related to damage in the brain caused by Alzheimer's disease can create speech impairments such asmumbling. In this case, formants do not change quickly and speech becomes less intelligible [23]. Changes in the first four formants are used as features in this research.

```python
# undetermined at this point
```

#### 6.5.2 Avg. abs. delta formant 2

```python
# undetermined at this point
```

#### 6.5.3 Avg. abs. delta formant 3

```python
# undetermined at this point
```

#### 6.5.4 Avg. abs. delta formant 4

```python
# undetermined at this point
```

#### 6.6 Voicing ratio
Another speech impairment is the loss of voicing in speech. In this case, the subject loses the ability to control the vibrations of the vocal cords, which results in breathy and noisy speech. The ratio of the total duration of voiced speech to the total duration of speech in the locution is used to detect any potential impairment in the vocal cords.

```python
# undetermined at this point
```

### Rate of speech-related features
#### 7.1 Phoneme rate
A basic identifier of rate of speech is the average number of phonemes spoken per second. The phoneme rate of a subject is computed by dividing the number of phonemes by the duration of the locution.

```python
# undetermined at this point
```

#### 7.2 Word rate
Similar to phoneme rate, word rate is used to measure the rate of speech at the word level. Word rate is computed by dividing the number of words by the duration of the locution.

```python
# undetermined at this point
```

## References
0. M Prince, M Guerchet, M Prina, World Alzheimer report 2013: Journey of caring: an analysis of long-term care for dementia. http://www.alz.co.uk/ research/world-report-2013 Accessed 2015-03-13
0. R Schmelzer, Roche Joins The Global CEO Initiative on Alzheimer's Disease. http://www.ceoalzheimersinitiative.org/node/71 Accessed 2014-08-20
0. MF Folstein, SE Folstein, PR McHugh, “Mini-mental state”: a practical method for grading the cognitive state of patients for the clinician. J. Psychiat. Res. 12(3), 189–198 (1975)
0. LM Bloudek, DE Spackman, M Blankenburg, SD Sullivan, Review and meta-analysis of biomarkers and diagnostic imaging in Alzheimer's disease. J. Alzheimer's Dis. 26(4), 627–645 (2011)
0. RS Bucks, S Singh, JM Cuerden, GK Wilcock, Analysis of spontaneous, conversational speech in dementia of Alzheimer type evaluation of an objective technique for analysing lexical performance. Aphasiology. 14(1), 71–91 (2000)
0. ET Prud'hommeaux, B Roark. Extraction of narrative recall patterns for neuropsychological assessment. Proceedings of the 12th Annual Conference of the International Speech Communication Association (Interspeech) (Florence, Italy, 2011), pp. 3021–3024
0. KC Fraser, JA Meltzer, NL Graham, C Leonard, G Hirst, SE Black, E Rochon, Automated classification of primary progressive aphasia subtypes from narrative speech transcripts. Cortex. 55, 43–60 (2014)
0. B Roark, M Mitchell, JP Hosom, K Hollingshead, J Kaye, Spoken language derived measures for detecting mild cognitive impairment. IEEE Trans. Audio Speech Lang. Process. 19(7), 2081–2090 (2011)
0. G Tosto, M Gasparini, GL Lenzi, G Bruno, Prosodic impairment in Alzheimer's disease: assessment and clinical relevance. J. Neuropsychiat. Clin. Neurosci. 23(2), 21–23 (2011)
0. DS Knopman, S Weintraub, VS Pankratz, Language and behavior domains enhance the value of the clinical dementia rating scale. 0zheimers Dement. 7(3), 293–299 (2011) 
0. SV Pakhomov, GE Smith, S Marino, A Birnbaum, N Graff-Radford, R Caselli, B Boeve, DS Knopman, A computerized technique to assess 0nguage use patterns in patients with frontotemporal dementia. J. Neurolinguistics. 23(2), 127–144 (2010)
0. V Iliadou, S Kaprinis, Clinical psychoacoustics in Alzheimer's disease central auditory processing disorders and speech 0terioration. Ann. Gen. Hosp. Psychiat. 2(1), 12 (2003)
0. I Hoffmann, D Nemeth, CD Dye, M Pakaski, T Irinyi, J Kalman, Temporal parameters of spontaneous speech in Alzheimer's disease. 0t. J. Speech Lang. Pathol. 12(1), 29–34 (2010)
0. SV Pakhomov, EA Kaiser, DL Boley, SE Marino, DS Knopman, AK Birnbaum, Effects of age and dementia on temporal cycles in 0ontaneous speech fluency. J. Neurolinguistics. 24(6), 619–635 (2011)
0. C Thomas, V Keselj, N Cercone, K Rockwood, E Asp, in Mechatronics and Automation 2005 IEEE International Conference. Automatic 0tection and rating of dementia of Alzheimer type through lexical analysis of spontaneous speech, vol. 3 (Niagara Falls, Canada, 005), pp. 1569–15743
0. LEE H, F Gayraud, F Hirsch, M Barkat-Defradas. Speech dysfluencies in normal and pathological aging: a comparison between 0zheimer patients and healthy elderly subjects. the 17th International Congress of Phonetic Sciences (ICPhS) (Hong Kong, China, 011), pp. 1174–1177
0. DA Snowdon, SJ Kemper, JA Mortimer, LH Greiner, DR Wekstein, WR Markesbery, Linguistic ability in early life and cognitive 0nction and Alzheimer's disease in late life. Findings from the Nun Study. JAMA. 275(7), 528–532 (1996)
0. K Oflazer, S Inkelas, in Proceedings of the EACLWorkshop on Finite State Methods in NLP. A finite state pronunciation lexicon for 0rkish, vol. 82 (Budapest, Hungary, 2003), pp. 900–918
0. K Oflazer, Two-level description of Turkish morphology. Literary Linguist. Comput. 9(2), 137–148 (1994)
0. v Brunet. Le Vocabulaire De Jean Giraudoux : Structure Et évolution : Statistique Et Informatique Appliquées à L'étude Des Textes 0Partir Des Données Du Trésor De La Langue Française. Le Vocabulaire des grands écrivains français (Genève, Slatkine, 1978). ASIN: 0000E99PZ
0. A Honore, Some simple measures of richness of vocabulary. Assoc. Literary Linguistic Comput. Bull. 7, 1979
0. D Biber, S Conrad, G Leech, The Longman student grammar of spoken and written English, (Harlow: Longman, 2002). ISBN: 0 582 237262
0. E Moore, MA Clements, JW Peifer, L Weisser, Critical analysis of the impact of glottal features in the classification of clinical 0pression in speech. Biomed. Eng. IEEE Trans. 55(1), 96–107 (2008)