from blabla.utils.global_params import *


def num_inflected_verbs(stanza_doc):
    """Returns the number of inflected verbs
		Args:
			None
		Returns
			num_inflected_verbs (int): The number of inflected verbs
	"""
    num_inflected_verbs = 0
    words = stanza_doc.sentences[0].words
    for word_idx in range(len(words) - 1):
        if (words[word_idx].pos == VERB) and (
            words[word_idx].text != words[word_idx].lemma
        ):
            num_inflected_verbs += 1
    return num_inflected_verbs


def num_gerund_verbs(stanza_doc):
    """The number of gerund verbs
		Args:
			None
		Returns:
			num_gerunds (int): the number of gerund verbs
	"""
    num_gerunds = 0
    for word in stanza_doc.sentences[0].words:
        if word.feats is not None:
            if (word.pos == VERB) and ('VerbForm=Ger' in word.feats):
                num_gerunds += 1
    return num_gerunds


def num_participle_verbs(stanza_doc):
    """The number of participle verbs
		Args:
			None
		Returns:
			num_participle_verbs (int): the number of participle verbs
	"""
    num_participle_verbs = 0
    for word in stanza_doc.sentences[0].words:
        if word.feats is not None:
            if (word.pos == VERB) and (
                'VerbForm=Part' in word.feats
            ):  # what if there are multiple words next to each other with Part?
                num_participle_verbs += 1
    return num_participle_verbs
