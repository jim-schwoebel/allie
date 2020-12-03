from blabla.sentence_processor.syntactic_feature_engine import *
from blabla.utils.global_params import *


class Num_Noun_Phrases(object):
    """Class to calculate the number of noun phrases
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the total number of noun phrases over all sentences
			Args:
				None
			Returns:
				The total number of noun phrases over all sentences
		"""
        tot_num_noun_phrases = 0
        for so in self.sentence_objs:
            tot_num_noun_phrases += num_noun_phrases(so.const_pt)
        return tot_num_noun_phrases


class Noun_Phrase_Rate(object):
    """Class to calculate the average number of noun phrases over all sentences
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the average number of noun phrases over all sentences
			Args:
				None
			Returns:
				The average number of noun phrases over all sentences
		"""
        tot_num_noun_phrases = 0
        for so in self.sentence_objs:
            tot_num_noun_phrases += num_noun_phrases(so.const_pt)
        return tot_num_noun_phrases / len(self.sentence_objs)


class Num_Verb_Phrases(object):
    """Class to calculate the number of verb phrases
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the total number of verb phrases over all sentences
			Args:
				None
			Returns:
				The total number of verb phrases over all sentences
		"""
        tot_num_verb_phrases = 0
        for so in self.sentence_objs:
            tot_num_verb_phrases += num_verb_phrases(so.const_pt)
        return tot_num_verb_phrases


class Verb_Phrase_Rate(object):
    """Class to calculate the average number of verb phrases over all sentences
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the average number of verb phrases over all sentences
			Args:
				None
			Returns:
				The average number of verb phrases over all sentences
		"""
        tot_num_verb_phrases = 0
        for so in self.sentence_objs:
            tot_num_verb_phrases += num_verb_phrases(so.const_pt)
        return tot_num_verb_phrases / len(self.sentence_objs)


class Num_Clauses(object):
    """Class to calculate the total number of clauses over all sentences
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the total number of clauses over all sentences
			Args:
				None
			Returns:
				The total number of clauses over all sentences
		"""
        tot_num_clauses = 0
        for so in self.sentence_objs:
            tot_num_clauses += num_clauses(so.const_pt)
        return tot_num_clauses


class Clause_Rate(object):
    """Class to calculate the number of clauses per sentences
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the average number of clauses over all sentences
			Args:
				None
			Returns:
				The average number of clauses over all sentences
		"""
        tot_num_clauses = 0
        for so in self.sentence_objs:
            tot_num_clauses += num_clauses(so.const_pt)
        return tot_num_clauses / len(self.sentence_objs)


class Num_Infinitive_Phrases(object):
    """Class to calculate the total number of infinitive phrases
		Note: This feature is available only for English
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the total number of infinitive phrases
			Args:
				None
			Returns:
				The total number of infinitive phrases over all sentences
		"""
        tot_num_inf_phrases = 0
        for so in self.sentence_objs:
            tot_num_inf_phrases += num_infinitive_phrases(so.stanza_doc)
        return tot_num_inf_phrases


class Infinitive_Phrase_Rate(object):
    """Class to calculate the number of infinitive phrases per sentence
		Note: This feature is available only for English
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the number of infinitive phrases per sentence
			Args:
				None
			Returns:
				The number of infinitive phrases per sentences
		"""
        tot_num_inf_phrases = 0
        for so in self.sentence_objs:
            tot_num_inf_phrases += num_infinitive_phrases(so.stanza_doc)
        return tot_num_inf_phrases / len(self.sentence_objs)


class Num_Dependent_Clauses(object):
    """Class to calculate the total number of dependent clauses
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the total number of dependent clauses
			Args:
				None
			Returns:
				The total number of dependent clauses
		"""
        tot_num_dep_clauses = 0
        for so in self.sentence_objs:
            tot_num_dep_clauses += num_dependent_clauses(so.const_pt)
        return tot_num_dep_clauses


class Dependent_Clause_Rate(object):
    """Class to calculate the number of dependent clauses per sentence
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the number of dependent clauses per sentence
			Args:
				None
			Returns:
				The number of dependent clauses per sentences
		"""
        tot_num_dep_clauses = 0
        for so in self.sentence_objs:
            tot_num_dep_clauses += num_dependent_clauses(so.const_pt)
        return tot_num_dep_clauses / len(self.sentence_objs)


class Num_Prepositional_Phrases(object):
    """Class to calculate the total number of prepositional phrases
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the total number of prepositional phrases
			Args:
				None
			Returns:
				The total number of prepositional phrases
		"""
        tot_num_prep_phrases = 0
        for so in self.sentence_objs:
            tot_num_prep_phrases += num_prepositional_phrases(so.const_pt)
        return tot_num_prep_phrases


class Prepositional_Phrase_Rate(object):
    """Class to calculate the number of prepositional phrases per sentence
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the number of prepositional phrases per sentence
			Args:
				None
			Returns:
				The number of prepositional phrases per sentence
		"""
        tot_num_prep_phrases = 0
        for so in self.sentence_objs:
            tot_num_prep_phrases += num_prepositional_phrases(so.const_pt)
        return tot_num_prep_phrases / len(self.sentence_objs)


class Prop_Nouns_With_Det(object):
    """Class to calculate the proportion of nouns with determiners
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the proportion of nouns that have a determiner as their dependency
			Args:
				None
			Returns:
				The number of nouns with determiners as their dependency
		"""
        num_nouns_with_determiners, num_nouns = 0, 0
        for so in self.sentence_objs:
            num_nouns_with_determiners += num_nouns_with_det(so.stanza_doc)
            num_nouns += so.pos_tag_counter.get_pos_tag_count(NOUN)
        if num_nouns != 0:
            return num_nouns_with_determiners / num_nouns
        return NOT_AVAILABLE


class Prop_Nouns_With_Adj(object):
    """Class to calculate the proportion of nouns with adjectives
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the proportion of nouns that have a adjective as their dependency
			Args:
				None
			Returns:
				The number of nouns with adjective as their dependency
		"""
        num_nouns_with_adjectives, num_nouns = 0, 0
        for so in self.sentence_objs:
            num_nouns_with_adjectives += num_nouns_with_adj(so.stanza_doc)
            num_nouns += so.pos_tag_counter.get_pos_tag_count(NOUN)
        if num_nouns != 0:
            return num_nouns_with_adjectives / num_nouns
        return NOT_AVAILABLE


class Max_Yngve_Depth(object):
    """Class to calculate the maximum Yngve depth averaged over all sentences
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the maximum Yngve depth averaged over all sentences
			Args:
				None
			Returns:
				The maximum Yngve depth averaged over all sentences
		"""
        total_max_yngve_depth = 0
        for so in self.sentence_objs:
            total_max_yngve_depth += max_yngve_depth(so.yngve_tree_root)
        num_sentences = len(self.sentence_objs)
        return total_max_yngve_depth / num_sentences


class Mean_Yngve_Depth(object):
    """Class to calculate the mean Yngve depth of each sentence, averaged over all sentences
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the mean Yngve depth of each sentence, averaged over all sentences
			Args:
				None
			Returns:
				The mean Yngve depth of each sentence, averaged over all sentences
		"""
        total_mean_yngve_depth = 0
        for so in self.sentence_objs:
            total_mean_yngve_depth += mean_yngve_depth(so.yngve_tree_root)
        num_sentences = len(self.sentence_objs)
        return total_mean_yngve_depth / num_sentences


class Total_Yngve_Depth(object):
    """Class to calculate the total Yngve depth of each sentence, averaged over all sentences
	"""

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
		"""
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the total Yngve depth of each sentence, averaged over all sentences
			Args:
				None
			Returns:
				The total Yngve depth of each sentence, averaged over all sentences
		"""
        total_all_yngve_depth = 0
        for so in self.sentence_objs:
            total_all_yngve_depth += total_yngve_depth(so.yngve_tree_root)
        num_sentences = len(self.sentence_objs)
        return total_all_yngve_depth / num_sentences


class Parse_Tree_Height(object):
    """Class to calculate the constituency parse tree height
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the average height of the constituency parse tree over all sentences
            Args:
                None
            Returns:
                The constituency parse tree height averaged over all sentences
        """
        tot_const_pt_height = 0
        for so in self.sentence_objs:
            tot_const_pt_height += const_pt_height(so.const_pt)
        return tot_const_pt_height / len(self.sentence_objs)


def syntactic_feature_processor(sentence_objs, feature, **kwArgs):
    """This method Returns the syntactic features across all the sentences depending on the type of feature requested
		Args:
			sentence_objs (list<Sentence>): a list of Sentence objects
			feature (str): a string name for the requested feature
		Returns:
			the feature value
	"""
    nr = globals()[feature.title()](sentence_objs)
    return nr.handle()
