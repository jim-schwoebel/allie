from blabla.utils.global_params import *
import blabla.utils.settings as settings
import numpy as np

def const_pt_height(const_pt):
    """The height of the constituency parse tree
    
		Args:
			const_pt (NLTK): The constituency parse tree

		Returns:
			height (int): the height of the constituency parse tree
	"""
    return const_pt.height()


def _leaves(const_pt, tag):
    """Returns the leaves of the constituency parse tree with a label

		Args:
			const_pt (NLTK): The constituency parse tree
            tag (str): The tag to be searched for in the parse tree

		Returns:
			leaves of the constituence parse trees (int): yields the leaves of the constituence parse tree matching a label
	"""
    for subtree in const_pt.subtrees(filter=lambda t: t.label() == tag):
        yield subtree


def num_prepositional_phrases(const_pt):
    """Returns the number of prepositional phrases

		Args:
			const_pt (NLTK): The constituency parse tree

		Returns
			number of prep phrases (int): number of prepositional phrases
	"""
    pp_chunks = []
    for leaf in _leaves(const_pt, PREPOSITIONAL_PHRASE):
        pp_chunks.append(leaf.leaves())
    return len(pp_chunks)


def num_verb_phrases(const_pt):
    """Returns the number of verb phrases
        Ref: For Chinese, please see this manual - https://repository.upenn.edu/cgi/viewcontent.cgi?article=1040&context=ircs_reports
        Ref: For English, please see this manual - https://www.cis.upenn.edu/~bies/manuals/root.pdf

    	Args:
    		const_pt (NLTK): The constituency parse tree

    	Returns
    		number of verb phrases (int): number of verb phrases
	"""
    vp_chunks = []
    vp_tag = None
    if settings.LANGUAGE in ["fr"]:
        lang = settings.LANGUAGE
    else:
        lang = "default"
    vp_tag = VERB_PHRASE_LANGUAGE_MAP[lang]
    for leaf in _leaves(const_pt, vp_tag):
        vp_chunks.append(leaf.leaves())
    return len(vp_chunks)


def num_noun_phrases(const_pt):
    """Returns the number of noun phrases

		Args:
			const_pt (NLTK): The constituency parse tree

		Returns
			number of noun phrases (int): number of noun phrases
	"""
    np_chunks = []
    for leaf in _leaves(const_pt, NOUN_PHRASE):
        np_chunks.append(leaf.leaves())
    return len(np_chunks)


def num_clauses(const_pt):
    # Ref: The PennTreeBank clauses list for English - http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
    # Ref: https://www.softschools.com/examples/grammar/phrases_and_clauses_examples/416/
    # Ref: Simple clauses in Chinese - http://prog3.com/sbdm/blog/cuixianpeng/article/details/16864785

    """Returns the total number of clauses
		Args:
			const_pt (NLTK): The constituency parse tree

		Returns
			tot_num_clauses (int): the total number of clauses
	"""
    clauses = []
    clause_tags = None
    if settings.LANGUAGE in ["fr", "zh-hant"]:
        lang = settings.LANGUAGE
    else:
        lang = "default"
    clause_tags = CLAUSES_LANGUAGE_MAP[lang]
    for clause_tag in clause_tags:
        for leaf in _leaves(const_pt, clause_tag):
            clauses.append(leaf.leaves())
    tot_num_clauses = len(clauses)
    return tot_num_clauses


def num_infinitive_phrases(stanza_doc):
    # Ref: https://www.grammar-monster.com/glossary/infinitive_phrase.html
    # NOTE: This feature is available only for English for now
    """Returns the number of infinitive phrases

		Args:
			stanza_doc (Stanza): The Stanza document object

		Returns
			num_inf_phrases (int): the number of infinitive phrases
	"""

    num_inf_phrases = 0
    for word in stanza_doc.sentences[0].words:
        if word.feats is not None:
            if "VerbForm=Inf" in word.feats:
                num_inf_phrases += 1
    return num_inf_phrases


def num_dependent_clauses(const_pt):
    # Ref: The PennTreeBank clauses list for English - http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
    # Ref: https://www.researchgate.net/figure/The-sample-constituency-parse-tree-S-simple-declarative-clause-NP-noun-phrase-VP_fig2_271223596
    # Ref: https://stackoverflow.com/questions/39320015/how-to-split-an-nlp-parse-tree-to-clauses-independent-and-subordinate
    # Ref: Depedent Clauses explanations: https://examples.yourdictionary.com/examples-of-dependent-clauses.html

    """Returns the number of dependent clauses
		Args:
			const_pt (NLTK): The constituency parse tree

		Returns
			number of dependent clauses (int): the number of dependent clauses
	"""
    dep_clauses = []
    clause_tags = None
    if settings.LANGUAGE in ["zh-hant", "fr"]:
        lang = settings.LANGUAGE
    else:
        lang = "default"
    clause_tags = SUBORD_CLAUSE_LANGUAGE_MAP[lang]
    for clause_tag in clause_tags:
        for leaf in _leaves(const_pt, clause_tag):
            dep_clauses.append(leaf.leaves())
    return len(dep_clauses)


def get_pos(word_id, words):
    for word in words:
        if int(word.id) == word_id:
            return word.pos


def num_nouns_with_det(stanza_doc):
    """Returns the number of nouns with determiners. This is done by counting the number of noun phrases that start with a determiner

        Args:
            stanza_doc (Stanza): The Stanza document object

        Returns
            num_nouns_with_det (int): The number of nouns with determiners
    """
    num_nouns_with_det = 0
    words = stanza_doc.sentences[0].words
    for word in words:
        if word.pos == DETERMINER:
            head_word_id = word.head
            if get_pos(head_word_id, words) == NOUN:
                num_nouns_with_det += 1
    return num_nouns_with_det


def num_nouns_with_adj(stanza_doc):
    """Returns the number of nouns with adjectives. This is done by counting the number of noun phrases that start with a adjective

        Args:
            stanza_doc (Stanza): The Stanza document object

        Returns
            num_nouns_with_adjectives (int): The number of nouns with adjectives
    """
    num_nouns_with_adjectives = 0
    words = stanza_doc.sentences[0].words
    for word in words:
        if word.pos == ADJECTIVE:
            head_word_id = word.head
            if get_pos(head_word_id, words) == NOUN:
                num_nouns_with_adjectives += 1
    return num_nouns_with_adjectives


def prop_nouns_with_det(stanza_doc, pos_tag_counter):
    """Returns the proportion of nouns with determiners. This is done by counting the number of noun phrases that start with a determiner (TODO)

		Args:
			stanza_doc (Stanza): The Stanza document object
            pos_tag_counter (obj): The POS tag counter object

		Returns
			prop of nouns with det (int): The proportion of nouns with determiners
	"""
    num_nouns = pos_tag_counter.get_pos_tag_count(NOUN)
    if num_nouns == 0:
        return NOT_AVAILABLE
    num_nouns_with_det = num_nouns_with_det(stanza_doc)
    return num_nouns_with_det / num_nouns


def max_yngve_depth(yngve_tree_root):
    """Returns the max depth of the ynvge tree of the sentence

		Args:
			yngve_tree_root (obj): The root node 

		Returns:
			int: The max depth of the yngve tree
	"""
    return max([leaf.score for leaf in yngve_tree_root.leaves])


def mean_yngve_depth(yngve_tree_root):
    """Returns the mean depth of the ynvge tree of the sentence

		Args:
			yngve_tree_root (obj): The root node 

		Returns:
			float: The mean depth of the yngve tree
	"""
    return np.mean([leaf.score for leaf in yngve_tree_root.leaves])


def total_yngve_depth(yngve_tree_root):
    """Returns the total depth of the ynvge tree of the sentence

		Args:
			yngve_tree_root (obj): The root node 

		Returns:
			int: The total depth of the yngve tree
	"""
    tot_score = 0
    for leaf in yngve_tree_root.leaves:
        tot_score += leaf.score
    return tot_score
