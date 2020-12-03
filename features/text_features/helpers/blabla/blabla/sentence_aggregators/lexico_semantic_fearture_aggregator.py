from blabla.sentence_processor.lexico_semantic_feature_engine import num_demonstratives
from blabla.utils.global_params import *
from collections import Counter
import numpy as np
import math
import blabla.utils.settings as settings
from blabla.utils.global_params import *


class Adjective_Rate(object):
    """Class to calculate the adjective rate
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the adjective rate
            Args:
                None
            Returns:
                The total number of adjectives to the total number of words
        """
        tot_num_adjs, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_adjs += so.pos_tag_counter.get_pos_tag_count(ADJECTIVE)
            tot_num_words += so.num_words()
        return tot_num_adjs / tot_num_words


class Adposition_Rate(object):
    """Class to calculate the adposition rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the noun rate
            Args:
                None
            Returns:
                The total number of nouns to the total number of words
        """
        tot_num_nouns, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_nouns += so.pos_tag_counter.get_pos_tag_count(ADPOSITION)
            tot_num_words += so.num_words()
        return tot_num_nouns / tot_num_words


class Adverb_Rate(object):
    """Class to calculate the adverb rate
        Ref: https://www.cs.toronto.edu/~kfraser/Fraser15-JAD.pdf
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the adverb rate
            Args:
                None
            Returns:
                The total number of adverbs to the total number of words
        """
        tot_num_advs, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_advs += so.pos_tag_counter.get_pos_tag_count(ADVERB)
            tot_num_words += so.num_words()
        return tot_num_advs / tot_num_words


class Auxiliary_Rate(object):
    """Class to calculate the auxiliary rate
        Ref: https://www.cs.toronto.edu/~kfraser/Fraser15-JAD.pdf
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the adverb rate
            Args:
                None
            Returns:
                The total number of adverbs to the total number of words
        """
        tot_num_advs, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_advs += so.pos_tag_counter.get_pos_tag_count(AUXILIARY)
            tot_num_words += so.num_words()
        return tot_num_advs / tot_num_words


class Conjunction_Rate(object):
    """Class to calculate the conjunctions rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the conjunctions rate
            Args:
                None
            Returns:
                The total number of conjunctions to the total number of words
        """
        tot_num_cconj, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_cconj += so.pos_tag_counter.get_pos_tag_count(CONJUNCTION)
            tot_num_words += so.num_words()
        return tot_num_cconj / tot_num_words


class Determiner_Rate(object):
    """Class to calculate the determiner rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the conjunctions rate
            Args:
                None
            Returns:
                The total number of conjunctions to the total number of words
        """
        tot_num_cconj, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_cconj += so.pos_tag_counter.get_pos_tag_count(DETERMINER)
            tot_num_words += so.num_words()
        return tot_num_cconj / tot_num_words


class Interjection_Rate(object):
    """Class to calculate the interjection rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the conjunctions rate
            Args:
                None
            Returns:
                The total number of conjunctions to the total number of words
        """
        tot_num_cconj, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_cconj += so.pos_tag_counter.get_pos_tag_count(INTERJECTION)
            tot_num_words += so.num_words()
        return tot_num_cconj / tot_num_words


class Noun_Rate(object):
    """Class to calculate the noun rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the noun rate
            Args:
                None
            Returns:
                The total number of nouns to the total number of words
        """
        tot_num_nouns, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_nouns += so.pos_tag_counter.get_pos_tag_count(NOUN)
            tot_num_words += so.num_words()
        return tot_num_nouns / tot_num_words


class Numeral_Rate(object):
    """Class to calculate the numeral rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the conjunctions rate
            Args:
                None
            Returns:
                The total number of conjunctions to the total number of words
        """
        tot_num_cconj, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_cconj += so.pos_tag_counter.get_pos_tag_count(NUMERAL)
            tot_num_words += so.num_words()
        return tot_num_cconj / tot_num_words


class Particle_Rate(object):
    """Class to calculate the particle rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the conjunctions rate
            Args:
                None
            Returns:
                The total number of conjunctions to the total number of words
        """
        tot_num_cconj, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_cconj += so.pos_tag_counter.get_pos_tag_count(PARTICLE)
            tot_num_words += so.num_words()
        return tot_num_cconj / tot_num_words


class Pronoun_Rate(object):
    """Class to calculate the pronoun rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the pronoun rate
            Args:
                None
            Returns:
                The total number of pronouns to the total number of words
        """
        tot_num_pron, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_pron += so.pos_tag_counter.get_pos_tag_count(PRONOUN)
            tot_num_words += so.num_words()
        return tot_num_pron / tot_num_words


class Proper_Noun_Rate(object):
    """Class to calculate the proper noun rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the pronoun rate
            Args:
                None
            Returns:
                The total number of pronouns to the total number of words
        """
        tot_num_pron, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_pron += so.pos_tag_counter.get_pos_tag_count(PROPER_NOUN)
            tot_num_words += so.num_words()
        return tot_num_pron / tot_num_words


class Punctuation_Rate(object):
    """Class to calculate the punctuation rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the pronoun rate
            Args:
                None
            Returns:
                The total number of pronouns to the total number of words
        """
        tot_num_pron, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_pron += so.pos_tag_counter.get_pos_tag_count(PUNCTUATION)
            tot_num_words += so.num_words()
        return tot_num_pron / tot_num_words


class Subordinating_Conjunction_Rate(object):
    """Class to calculate the subordinating conjuction rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the pronoun rate
            Args:
                None
            Returns:
                The total number of pronouns to the total number of words
        """
        tot_num_pron, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_pron += so.pos_tag_counter.get_pos_tag_count(SUBORDINATING_CONJUNCTION)
            tot_num_words += so.num_words()
        return tot_num_pron / tot_num_words


class Symbol_Rate(object):
    """Class to calculate the symbol rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the pronoun rate
            Args:
                None
            Returns:
                The total number of pronouns to the total number of words
        """
        tot_num_pron, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_pron += so.pos_tag_counter.get_pos_tag_count(SYMBOL)
            tot_num_words += so.num_words()
        return tot_num_pron / tot_num_words


class Verb_Rate(object):
    """Class to calculate the verb rate
        Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the verb rate
            Args:
                None
            Returns:
                The total number of verbs to the total number of words
        """
        tot_num_verbs, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_verbs += so.pos_tag_counter.get_pos_tag_count(VERB)
            tot_num_words += so.num_words()
        return tot_num_verbs / tot_num_words


class Demonstrative_Rate(object):
    """Class to calculate the demonstratives rate
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the verb rate
            Args:
                None
            Returns:
                The total number of demonstratives to the total number of words
        """
        tot_num_demons, tot_num_words = 0, 0
        for so in self.sentence_objs:
            tot_num_demons += num_demonstratives(so.stanza_doc)
            tot_num_words += so.num_words()
        return tot_num_demons / tot_num_words


class Possessive_Rate(object):
    """Class to calculate the possessive rate
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3642700/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the possessive rate
            Args:
                None
            Returns:
                The total number of adjectives and pronouns to the total number of words
        """
        tot_num_adjs, tot_num_pron, tot_num_words = 0, 0, 0
        for so in self.sentence_objs:
            tot_num_adjs += so.pos_tag_counter.get_pos_tag_count(ADJECTIVE)
            tot_num_pron += so.pos_tag_counter.get_pos_tag_count(PRONOUN)
            tot_num_words += so.num_words()
        return (tot_num_adjs + tot_num_pron) / tot_num_words


class Noun_Verb_Ratio(object):
    """Class to calculate the ratio of the number of nouns to the number of verbs
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the noun to verb
            Args:
                None
            Returns:
                The total number of nouns to the number of verbs
        """
        tot_num_nouns, tot_num_verbs = 0, 0
        for so in self.sentence_objs:
            tot_num_nouns += so.pos_tag_counter.get_pos_tag_count(NOUN)
            tot_num_verbs += so.pos_tag_counter.get_pos_tag_count(VERB)
        if tot_num_verbs != 0:
            return tot_num_nouns / tot_num_verbs
        return NOT_AVAILABLE


class Noun_Ratio(object):
    """Class to calculate the ratio of the number of nouns to the total number of nouns and verbs
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the noun ratio
            Args:
                None
            Returns:
                The total number of nouns to the total number of nouns and verbs
        """
        tot_num_nouns, tot_num_verbs = 0, 0
        for so in self.sentence_objs:
            tot_num_nouns += so.pos_tag_counter.get_pos_tag_count(NOUN)
            tot_num_verbs += so.pos_tag_counter.get_pos_tag_count(VERB)
        if (tot_num_nouns + tot_num_verbs) != 0:
            return tot_num_nouns / (tot_num_nouns + tot_num_verbs)


class Pronoun_Noun_Ratio(object):
    """Class to calculate the ratio of the number of pronouns to the total number of nouns
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the pronoun to noun ratio
            Args:
                None
            Returns:
                The ratio of the total number of pronouns to the number of nouns
        """
        tot_num_prons, tot_num_nouns = 0, 0
        for so in self.sentence_objs:
            tot_num_prons += so.pos_tag_counter.get_pos_tag_count(PRONOUN)
            tot_num_nouns += so.pos_tag_counter.get_pos_tag_count(NOUN)
        if tot_num_nouns != 0:
            return tot_num_prons / tot_num_nouns
        return NOT_AVAILABLE

class Total_Dependency_Distance(object):
    """Class to calculate the sum of dependency distances
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the total dependency distance across all sentences
            Args:
                None
            Returns:
                the sum of dependency distances
        """
        tot_dist = 0
        for so in self.sentence_objs:
            sd = so.stanza_doc.to_dict()[0]
            tot_dist += np.sum([abs(int(dep['id']) - dep['head']) for dep in sd])
        return tot_dist

class Average_Dependency_Distance(object):
    """Class to calculate the sum of dependency distances
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the total dependency distance across all sentences
            Args:
                None
            Returns:
                the sum of dependency distances
        """
        tot_dist = []
        for so in self.sentence_objs:
            sd = so.stanza_doc.to_dict()[0]
            tot_dist.append(sum([abs(int(dep['id']) - dep['head']) for dep in sd]))

        if tot_dist:
            return np.mean(tot_dist)
        return NOT_AVAILABLE


class Total_Dependencies(object):
    """Class to calculate the number of unique syntactic dependencies
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the total number of unique dependencies across sentences
            Args:
                None
            Returns:
                the total number of unique dependencies
        """
        deprels = []
        for so in self.sentence_objs:
            sd = so.stanza_doc.to_dict()[0]
            deprels.extend([dep['deprel'] for dep in sd])
        return len(set(deprels))


class Average_Dependencies(object):
    """Class to calculate the average number of unique syntactic dependencies
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the average number of unique dependencies across sentences
            Args:
                None
            Returns:
                the average number of unique dependencies
        """
        num_deprels = []
        for so in self.sentence_objs:
            sd = so.stanza_doc.to_dict()[0]
            deprels = set([dep['deprel'] for dep in sd])
            num_deprels.append(len(deprels))

        if num_deprels:
            return np.mean(num_deprels)
        return NOT_AVAILABLE


class Closed_Class_Word_Rate(object):
    """Class to calculate the proportion of closed class words
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the proportion of close class words
            Args:
                None
            Returns:
                The ratio of the total number of determiners, prepositions, pronouns and conjunctions to the total number of words
        """
        tot_num_det, tot_num_prep, tot_num_pron, tot_num_cconj, tot_num_words = (
            0,
            0,
            0,
            0,
            0,
        )
        for so in self.sentence_objs:
            tot_num_det += so.pos_tag_counter.get_pos_tag_count(DETERMINER)
            tot_num_prep += so.pos_tag_counter.get_pos_tag_count(ADPOSITION)
            tot_num_pron += so.pos_tag_counter.get_pos_tag_count(PRONOUN)
            tot_num_cconj += so.pos_tag_counter.get_pos_tag_count(CONJUNCTION)
            tot_num_words += so.num_words()
        return (
            tot_num_det + tot_num_prep + tot_num_pron + tot_num_cconj
        ) / tot_num_words


class Open_Class_Word_Rate(object):
    """Class to calculate the proportion of open class word_count
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the proportion of open class words
            Args:
                None
            Returns:
                The ratio of the total number of nouns, verbs, adjectives and adverbs to the total number of words
        """
        tot_num_nouns, tot_num_verbs, tot_num_adjs, tot_num_advs, tot_num_words = (
            0,
            0,
            0,
            0,
            0,
        )
        for so in self.sentence_objs:
            tot_num_nouns += so.pos_tag_counter.get_pos_tag_count(NOUN)
            tot_num_verbs += so.pos_tag_counter.get_pos_tag_count(VERB)
            tot_num_adjs += so.pos_tag_counter.get_pos_tag_count(ADJECTIVE)
            tot_num_advs += so.pos_tag_counter.get_pos_tag_count(ADVERB)
            tot_num_words += so.num_words()
        return (
            tot_num_nouns + tot_num_verbs + tot_num_adjs + tot_num_advs
        ) / tot_num_words


class Content_Density(object):
    """Class to calculate the content density of words
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the content density of words
            Args:
                None
            Returns:
                The ratio of the total number of open class words to the total number of closed class words
        """
        tot_num_nouns, tot_num_verbs, tot_num_adjs, tot_num_advs = 0, 0, 0, 0
        tot_num_det, tot_num_prep, tot_num_pron, tot_num_cconj = 0, 0, 0, 0
        for so in self.sentence_objs:
            tot_num_nouns += so.pos_tag_counter.get_pos_tag_count(NOUN)
            tot_num_verbs += so.pos_tag_counter.get_pos_tag_count(VERB)
            tot_num_adjs += so.pos_tag_counter.get_pos_tag_count(ADJECTIVE)
            tot_num_advs += so.pos_tag_counter.get_pos_tag_count(ADVERB)
        for so in self.sentence_objs:
            tot_num_det += so.pos_tag_counter.get_pos_tag_count(DETERMINER)
            tot_num_prep += so.pos_tag_counter.get_pos_tag_count(ADPOSITION)
            tot_num_pron += so.pos_tag_counter.get_pos_tag_count(PRONOUN)
            tot_num_cconj += so.pos_tag_counter.get_pos_tag_count(CONJUNCTION)
        numerator = tot_num_nouns + tot_num_verbs + tot_num_adjs + tot_num_advs
        denominator = tot_num_det + tot_num_prep + tot_num_pron + tot_num_cconj
        if denominator == 0:
            return NOT_AVAILABLE
        return numerator / denominator


class Idea_Density(object):
    """Class to calculate the idea density of words
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the idea density of words
            Args:
                None
            Returns:
                The ratio of the total number of verbs, adjectives, adverbs, prepositions, conjunctions to the number of words
        """
        (
            tot_num_verbs,
            tot_num_adjs,
            tot_num_advs,
            tot_num_preps,
            tot_num_cconjs,
            tot_num_words,
        ) = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
        for so in self.sentence_objs:
            tot_num_verbs += so.pos_tag_counter.get_pos_tag_count(VERB)
            tot_num_adjs += so.pos_tag_counter.get_pos_tag_count(ADJECTIVE)
            tot_num_advs += so.pos_tag_counter.get_pos_tag_count(ADVERB)
            tot_num_preps += so.pos_tag_counter.get_pos_tag_count(ADPOSITION)
            tot_num_cconjs += so.pos_tag_counter.get_pos_tag_count(CONJUNCTION)
            tot_num_words += so.num_words()
        return (
            tot_num_verbs + tot_num_adjs + tot_num_advs + tot_num_preps + tot_num_cconjs
        ) / tot_num_words


class Honore_Statistic(object):
    """Class to calculate the honore's statistic
        Ref: https://www.aclweb.org/anthology/W16-1902.pdf
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the honore's statistic
            Args:
                None
            Returns:
                The honore's statistic of the words
        """
        all_words = []
        num_unique_words_spoken, num_words_spoken_only_once = 0, 0
        for so in self.sentence_objs:
            all_words.extend([word.text for word in so.stanza_doc.sentences[0].words])

        num_unique_words_spoken = len(set(all_words))
        word_counts = dict(Counter(all_words))
        for key, val in word_counts.items():
            if val == 1:
                num_words_spoken_only_once += 1
        num_words = len(all_words)
        if (num_words_spoken_only_once == num_unique_words_spoken) or (num_unique_words_spoken == 0) or (num_words == 0):
            return NOT_AVAILABLE
        honore_statistic = (100 * math.log(num_words)) / (
            1 - (num_words_spoken_only_once) / (num_unique_words_spoken)
        )
        return honore_statistic


class Brunet_Index(object):
    """Class to calculate the brunet's statistic
        Ref: https://www.aclweb.org/anthology/W16-1902.pdf
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the brunet's statistic
            Args:
                None
            Returns:
                The brunet's statistic of the words
        """
        num_unique_words_spoken = 0
        all_words = []
        for so in self.sentence_objs:
            all_words.extend([word.text for word in so.stanza_doc.sentences[0].words])
        num_unique_words_spoken = len(set(all_words))
        num_words = len(all_words)
        brunet_index = math.pow(num_words, math.pow(num_unique_words_spoken, -0.165))
        return brunet_index


class Type_Token_Ratio(object):
    """Class to calculate the type-token ratio
        Ref: https://www.tandfonline.com/doi/abs/10.1080/02687038.2017.1303441
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the type-token statistic
            Args:
                None
            Returns:
                The ratio of the number of word types to the number of words
        """
        all_words = []
        all_word_lemmas = []
        for so in self.sentence_objs:
            all_words.extend([word.text for word in so.stanza_doc.sentences[0].words])
            all_word_lemmas.extend(
                [word.lemma for word in so.stanza_doc.sentences[0].words]
            )
        num_word_types = len(set(all_word_lemmas))
        num_words = len(all_words)
        return num_word_types / num_words


class Word_Length(object):
    """Class to calculate the mean word length
        Ref: https://pubmed.ncbi.nlm.nih.gov/26484921/
    """

    def __init__(self, sentence_objs):
        """The init method to initialize with an array of sentence objects
        """
        self.sentence_objs = sentence_objs

    def handle(self):
        """Method to calculcate the mean word length
            Args:
                None
            Returns:
                The mean length of the word across all sentences
        """
        all_words = []
        for so in self.sentence_objs:
            all_words.extend([word.text for word in so.stanza_doc.sentences[0].words])
        mean_word_length = np.mean([len(word) for word in all_words])
        return mean_word_length


def lexico_semantic_feature_processor(sentence_objs, feature, **kwArgs):
    """This method Returns the lexico semantic features across all the sentences depending on the type of feature requested
		Args:
			sentence_objs (list<Sentence>): a list of Sentence objects
			feature (str): a string name for the requested feature
		Returns:
			the feature value
	"""
    nr = globals()[feature.title()](sentence_objs)
    return nr.handle()
