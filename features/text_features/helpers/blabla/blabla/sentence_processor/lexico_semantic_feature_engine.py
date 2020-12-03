from collections import Counter
import math
from blabla.utils.global_params import *
from blabla.utils import *


def num_demonstratives(stanza_doc):
    """The number of demonstravives

            Args:
                stanza_doc (nltk.Tree): The dependency parse tree

            Returns:
                (int): the number of demonstravives
        """
    return len([1 for word in stanza_doc.sentences[0].words if ((word.feats is not None) and ('PronType=Dem' in word.feats))])


def num_unique_words(stanza_doc):
    """Returns the number of unique words

        Args:
            stanza_doc (nltk.Tree): The dependency parse tree

        Returns:
            number of unique words 
    """
    return len(set([word.text for word in stanza_doc.sentences[0].words]))


def num_word_types(stanza_doc):
    """Returns the number of word types

        Args:
            stanza_doc (nltk.Tree): The dependency parse tree

        Returns:
            number of word types
    """
    return len(set([word.lemma for word in stanza_doc.sentences[0].words]))


def compute_mean_word_length(stanza_doc):
    """Returns the mean word length

            Args:
                stanza_doc (nltk.Tree): The dependency parse tree
                
            Returns:
                mean length of all words in the sentence
        """
    return np.mean([len(word.text) for word in stanza_doc.sentences[0].words])
