from blabla.sentence_processor.morpho_syntactic_feature_engine import (
    num_inflected_verbs,
    num_gerund_verbs,
    num_participle_verbs,
)
from blabla.utils.global_params import *


class Prop_Inflected_Verbs(object):
    """Class to calculcate the proportion of inflected verbs
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the proportion of inflected verbs
            Args:
                None
            Returns:
                The proportion of inflected verbs
        """
        tot_num_inflected_verbs, tot_num_verbs = 0, 0
        for so in self.sentence_objs:
            tot_num_verbs += so.pos_tag_counter.get_pos_tag_count(VERB)
            tot_num_inflected_verbs += num_inflected_verbs(so.stanza_doc)
        if tot_num_verbs != 0:
            return tot_num_inflected_verbs / tot_num_verbs
        return NOT_AVAILABLE


class Prop_Auxiliary_Verbs(object):
    """Class to calculcate the proportion of auxiliary verbs
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculate the proportion of inflected verbs
            Args:
                None
            Returns:
                The proportion of auxiliary verbs
        """
        tot_num_auxiliary_verbs, tot_num_verbs = 0, 0
        for so in self.sentence_objs:
            tot_num_auxiliary_verbs += so.pos_tag_counter.get_pos_tag_count(AUXILIARY)
            tot_num_verbs += so.pos_tag_counter.get_pos_tag_count(VERB)
        if tot_num_verbs != 0:
            return tot_num_auxiliary_verbs / tot_num_verbs
        return NOT_AVAILABLE


class Prop_Gerund_Verbs(object):
    """Class to calculcate the proportion of gerund verbs
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the proportion of gerund verbs
            Args:
                None
            Returns:
                The proportion of gerund verbs
        """
        tot_num_gerunds, tot_num_verbs = 0, 0
        for so in self.sentence_objs:
            tot_num_gerunds += num_gerund_verbs(so.stanza_doc)
            tot_num_verbs += so.pos_tag_counter.get_pos_tag_count(VERB)
        if tot_num_verbs != 0:
            return tot_num_gerunds / tot_num_verbs
        return NOT_AVAILABLE


class Prop_Participles(object):
    """Class to calculcate the proportion of participle verbs
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the proportion of participle verbs
            Args:
                None
            Returns:
                The proportion of participle verbs
        """
        tot_num_participle_verbs, tot_num_verbs = 0, 0
        for so in self.sentence_objs:
            tot_num_participle_verbs += num_participle_verbs(so.stanza_doc)
            tot_num_verbs += so.pos_tag_counter.get_pos_tag_count(VERB)
        if tot_num_verbs != 0:
            return tot_num_participle_verbs / tot_num_verbs
        return NOT_AVAILABLE


def morpho_syntactic_feature_processor(sentence_objs, feature, **kwArgs):
    """This method Returns the morpho syntactic features across all the sentences depending on the type of feature requested
		Args:
			sentence_objs (list<Sentence>): a list of Sentence objects
			feature (str): a string name for the requested feature
		Returns:
			the feature value
	"""
    nr = globals()[feature.title()](sentence_objs)
    return nr.handle()
