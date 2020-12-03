def num_discourse_markers(stanza_doc):
    # Ref: https://universaldependencies.org/docsv1/u/dep/all.html#al-u-dep/discourse
    """Returns the number of discourse markers

        Args:
            stanza_doc (obj): The stanza document object
        Returns
            (int): the number of discourse markers
    """
    return len([1 for word in stanza_doc.sentences[0].words if word.deprel == 'discourse'])
