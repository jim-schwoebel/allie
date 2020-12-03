from blabla.sentence_processor.discourse_and_pragmatic_feature_engine import (
    num_discourse_markers,
)


class Num_Discourse_Markers(object):
    """Class to calculate the total number of discourse markers
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the total number of discourse markers
            Args:
                None
            Returns:
                tot_num_discourse_markers (int): The total number of discourse markers
        """
        tot_num_discourse_markers = 0
        for so in self.sentence_objs:
            tot_num_discourse_markers += num_discourse_markers(so.stanza_doc)
        return tot_num_discourse_markers


class Discourse_Marker_Rate(object):
    """Class to calculate the number of discourse markers per sentence
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the number of discourse markers per sentence
            Args:
                None
            Returns:
                The number of discourse markers per sentence
        """
        tot_num_discourse_markers = 0
        for so in self.sentence_objs:
            tot_num_discourse_markers += num_discourse_markers(so.stanza_doc)
        return tot_num_discourse_markers / len(self.sentence_objs)


def discourse_and_pragmatic_feature_processor(sentence_objs, feature, **kwArgs):
    """Extract discourse and pragmatic features across all sentence objects
		Args:
			sentence_objs (list<Sentence>): a list of Sentence objects
			feature (str): a string name for the requested feature
		Returns:
			the feature value
	"""
    nr = globals()[feature.title()](sentence_objs)
    return nr.handle()
