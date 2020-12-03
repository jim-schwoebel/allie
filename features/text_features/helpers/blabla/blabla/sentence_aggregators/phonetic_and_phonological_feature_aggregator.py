import blabla.sentence_aggregators.phonetic_features_settings as pfs
from blabla.utils.global_params import *
from blabla.utils.exceptions import *
import blabla.utils.settings as settings
import numpy as np


class Word(object):
    """Class to represent a word"""

    def __init__(self, word):
        self.start_time = word['start_time']
        self.end_time = word['start_time'] + word['duration']
        self.duration = word['duration']

class Maximum_Speech_Rate(object):
    """Class to calculate the maximum speech rate
    """

    def __init__(self, sentence_objs, **kwArgs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs
        self.num_rapid_sentences = kwArgs.get(
            'num_rapid_sentences', pfs.NUMBER_OF_RAPID_SENTENCES
        )

    def handle(self):
        """Method to calculate the maximum speech rate
            Args:
                None
            Returns:
                (float): The maximum speech rate
        """
        words = []
        window_size = 10
        num_speech_rates = 3
        # store all words as a sequence in a list as objects
        for so in self.sentence_objs:
            for word_tok in so.json['words']:
                words.append(Word(word_tok))

        # If we have less than 12 words in total, then we cannot calculate this feature
        num_words = len(words)
        if num_words < (window_size + num_speech_rates - 1):
            return NOT_AVAILABLE

        # calculate the speech rates
        speech_rates = []
        for idx in range(0, num_words - window_size):
            start_time, end_time = words[idx].start_time, words[idx+window_size].end_time
            speech_rates.append((window_size / (end_time - start_time)) * 60.0)

        # take the mean of the highest 3 speech rates
        return np.mean(sorted(speech_rates, reverse=True)[:num_speech_rates])


class Num_Pauses(object):
    """Class to calculate the total number of pauses
		Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
	"""

    def __init__(self, sentence_objs, **kwArgs):
        """The init method to initialize with an array of sentence objects and the optional pause duration
		"""
        self.sentence_objs = sentence_objs
        self.pause_duration = kwArgs.get('pause_duration', pfs.PAUSE_DURATION)

    def handle(self):
        """Method to calculate the total number of pauses
			Args: 
				None:
			Returns:
				total_num_pauses (float): total number of pauses across all sentences between words
		"""
        tot_num_pauses = 0
        for so in self.sentence_objs:
            tot_num_pauses += so.num_pauses(self.pause_duration)
        return tot_num_pauses


class Total_Pause_Time(object):
    """Class to calculate the total pause time
		Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
	"""

    def __init__(self, sentence_objs, **kwArgs):
        """The init method to initialize with an array of sentence objects and the optional pause duration
		"""
        self.sentence_objs = sentence_objs
        self.pause_duration = kwArgs.get('pause_duration', pfs.PAUSE_DURATION)

    def handle(self):
        """Method to calculate the total pause time
			Args:
				None:
			Returns:
				total_pause_time (float): total pause time between words across all sentences
		"""
        total_pause_time = 0.0
        for so in self.sentence_objs:
            total_pause_time += so.tot_pause_time(self.pause_duration)
        return total_pause_time


class Mean_Pause_Duration(object):
    """Class to calculate the mean pause duration 
		Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
	"""

    def __init__(self, sentence_objs, **kwArgs):
        """The init method to initialize with an array of sentence objects and the optional pause duration
		"""
        self.sentence_objs = sentence_objs
        self.pause_duration = kwArgs.get('pause_duration', pfs.PAUSE_DURATION)

    def handle(self):
        """Method to calculate the mean pause duration
			Args:
				None
			Returns:
				(float): The mean pause duration across all sentences between words
		"""
        tot_pause_duration = 0.0
        tot_num_pauses = 0
        for so in self.sentence_objs:
            tot_pause_duration += so.tot_pause_time(self.pause_duration)
            tot_num_pauses += so.num_pauses(self.pause_duration)
        return tot_pause_duration / tot_num_pauses


class Between_Utterance_Pause_Duration(object):
    """Class to calculate the between utterance pause duration
		Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
	"""

    def __init__(self, sentence_objs, **kwArgs):
        """The init method to initialize with an array of sentence objects and the optional pause between utterance duration
		"""
        self.sentence_objs = sentence_objs
        self.pause_between_utterance_duration = kwArgs.get(
            'pause_between_utterance_duration', pfs.PAUSE_BETWEEN_UTTERANCE_DURATION
        )

    def handle(self):
        """Method to calculate the average between utterance pause duration
			Args:
				None
			Returns:
				(float): The average between utterance pause duration
		"""
        pause_durations = []
        for prev_sent, next_sent in zip(self.sentence_objs[:-1], self.sentence_objs[1:]):
            duration = next_sent.start_time - prev_sent.end_time
            if (duration >= self.pause_between_utterance_duration):
                pause_durations.append(duration)
        if len(pause_durations) == 0:
            # in case there is no pause between sentences at all
            return NOT_AVAILABLE
        return np.mean(pause_durations)


class Hesitation_Ratio(object):
    """Class to calculate the total hesitation ratio
		Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
	"""

    def __init__(self, sentence_objs, **kwArgs):
        """The init method to initialize with an array of sentence objects and the optional pause duration value for hesitation
		"""
        self.sentence_objs = sentence_objs
        self.pause_duration = kwArgs.get(
            'pause_duration_for_hesitation', pfs.PAUSE_DURATION_FOR_HESITATION
        )

    def handle(self):
        """Method to calculate the hesitation ratio
    		Args:
    			None
    		Returns:
    			(float): The ratio of the total duration of hesitation to the total speech time
    	"""
        tot_hesitation_duration, tot_speech_time = 0.0, 0.0
        for so in self.sentence_objs:
            tot_speech_time += so.speech_time
        for so in self.sentence_objs:
            tot_hesitation_duration += so.tot_pause_time(self.pause_duration)
        return tot_hesitation_duration / tot_speech_time


class Speech_Rate(object):
    """Class to calculate the number of words spoken per minute
		Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
	"""

    def __init__(self, sentence_objs, **kwArgs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculate the number of words per minute
    		Args:
    			None
    		Returns:
    			(float): The number of words per minute
    	"""
        tot_num_words, tot_locution_time = 0, 0
        for so in self.sentence_objs:
            tot_num_words += so.num_words()
            tot_locution_time += so.locution_time
        return (tot_num_words / tot_locution_time) * 60.0


class Total_Phonation_Time(object):
    """Class to calculate the total phonation time
    """

    def __init__(self, sentence_objs, **kwArgs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculate the total phonation time
            Args:
                None
            Returns:
                tot_speech_time (float): The total phonation time
        """
        tot_speech_time = 0
        for so in self.sentence_objs:
            tot_speech_time += so.speech_time
        return tot_speech_time


class Standardized_Phonation_Time(object):
    """Class to calculate the standardized phonation rate
    """

    def __init__(self, sentence_objs, **kwArgs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculate the standardized phonation rate
            Args:
                None
            Returns:
                (float): The standardized phonation rate
        """
        tot_num_words = 0
        for so in self.sentence_objs:
            tot_num_words += so.num_words()
        return tot_num_words / Total_Phonation_Time(self.sentence_objs).handle()


class Total_Locution_Time(object):
    """Class to calculate the total locution time
		Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
	"""

    def __init__(self, sentence_objs, **kwArgs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculate the total locution time
    		Args:
    			None
    		Returns:
    			float: The amount of time in the sample containing both speech and pauses    			
    	"""
        return self.sentence_objs[-1].end_time - self.sentence_objs[0].start_time


def phonetic_and_phonological_feature_processor(sentence_objs, feature, **kwArgs):
    """This method Returns the phonetic and phonological features across all the sentences depending on the type of feature requested
		Args:
			sentence_objs (list<Sentence>): a list of Sentence objects
			feature (str): a string name for the requested feature
            kwArgs (list): A list of optional arguments for different features
		Returns:
			the feature value
	"""
    if settings.INPUT_FORMAT != "json":
        raise InvalidFeatureException(
            'You have requested the feature {} that required time stamps for the words. Please check the input format'.format(
                feature
            )
        )
    nr = globals()[feature.title()](sentence_objs, **kwArgs)
    return nr.handle()
