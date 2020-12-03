import stanza


class PosTagCounter(object):
    """The class that counts the number of pos tags of various types in a sentence
    """

    def __init__(self, stanza_doc):
        """The initialization method that take a dependency parse tree as input
            Args:
                stanza_doc (nltk.Tree): the dependency parse tree
            Returns:
                None
        """
        self.stanza_doc = stanza_doc

    def get_pos_tag_count(self, pos_tag):
        """Returns the number of nouns
            Args:
                None
            Returns:
                number of nouns in the sentence
        """
        return len(
            [
                word
                for word in self.stanza_doc.sentences[0].words
                if (word.pos == pos_tag)
            ]
        )
