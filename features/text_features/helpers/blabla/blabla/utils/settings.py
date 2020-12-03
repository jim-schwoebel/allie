def init():
    global INPUT_FORMAT
    global LANGUAGE
    global CONST_PT_FEATURES
    INPUT_FORMAT = None
    LANGUAGE = None
    CONST_PT_FEATURES = [
        "parse_tree_height",
        "num_noun_phrases",
        "noun_phrase_rate",
        "num_verb_phrases",
        "verb_phrase_rate",
        "num_infinitive_phrases",
        "infinitive_phrase_rate",
        "num_prepositional_phrases",
        "prepositional_phrase_rate",
        "max_yngve_depth",
        "mean_yngve_depth",
        "total_yngve_depth",
        "num_clauses",
        "clause_rate",
        "num_dependent_clauses",
        "dependent_clause_rate"
    ]
