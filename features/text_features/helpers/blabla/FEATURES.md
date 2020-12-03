The below table lists the different features in BlaBla. In the table, we provide the following columns of information for all the features:

* ***feature name*** - The name of the feature
* ***feature description*** - A short description on what the feature means
* ***feature key*** - The key that you need in the `compute_features` method
* ***core (# langs)*** - The base framework that BlaBla uses (Stanza / CoreNLP) for feature extraction
* ***requires Stanza*** - (Yes/No) depending on whether this features requires Stanza
* ***requires CoreNLP*** - (Yes/No) depending on whether this features requires CoreNLP
* ***input format*** - The actual format of the input supported. It is either String (free text) or JSON
* ***default param name*** - Few of the features allow the user to override a default value such as `pause_duration` between words. This columns tells you the additional key you can provide in the `compute_features` method for the feature
* ***default param value*** - This is the default value taken for the parameter relevant to the feature

feature name | feature description | feature key | core (#langs) | requires Stanza | requires CoreNLP | input format | default param name | default param value
------------ | ------------- | ------------ | ------------- | ------------ | ------------- | ------------ | ------------ | ------------
number of pauses| The total number of pauses between words greater than a threshold | ***num_pauses*** | Stanza(66) | No | No | JSON | ***pause_duration*** | 0.35
total pause time| The total duration of pauses between words greater than a threshold | ***total_pause_time*** | Stanza(66) | No | No | JSON | ***pause_duration*** | 0.35
mean pause duration| The average duration of all pauses between words greater than a threshold | ***mean_pause_duration*** | Stanza(66) | No | No | JSON | ***pause_duration*** | 0.35
between utterance pause duration| The proportion of total druation of pauses that are between utterances | ***between_utterance_pause_duration*** | Stanza(66) | No | No | JSON | ***pause_between_utterance_duration*** | 0.035
hesitation ratio| The average gap between sentences | ***hesitation_ratio*** | Stanza(66) | No | No | JSON | ***pause_duration_for_hesitation*** | 0.030
speech rate| The number of words per minute | ***speech_rate*** | Stanza(66) | No | No | JSON | - | -
maximum speech rate| The average number of words per minute across the top N (set to 10 by default) rapid sentences| ***maximum_speech_rate*** | Stanza(66) | No | No | JSON | ***num_rapid_sentences*** | 10
total phonation time| Total time duration of all words across all sentences | ***total_phonation_time*** | Stanza(66) | No | No | JSON | - | -
standardized phonation time| The total number of words divided by the total phonation time | ***standardized_phonation_time*** | Stanza(66) | No | No | JSON | - | -
total locution time| The total amount of time in speech that contains both speech and pauses | ***total_locution_time*** | Stanza(66) | No | No | JSON | - | -
noun rate|The rate of nouns across sentences |***noun_rate***|Stanza(66)|Yes|No|String or JSON | - | -
verb rate|The rate of verbs across sentences|***verb_rate***|Stanza(66)|Yes|No|String or JSON | - | -
demonstrative rate|The rate of demonstrative across sentences|***demonstrative_rate***|Stanza(66)|Yes|No|String or JSON | - | -
adjective rate|The rate of adjectives across sentences|***adjective_rate***|Stanza(66)|Yes|No|String or JSON | - | -
adposition rate|The rate of adpositions across sentences|***adposition_rate***|Stanza(66)|Yes|No|String or JSON | - | -
adverb rate|The rate of adverbs across sentences|***adverb_rate***|Stanza(66)|Yes|No|String or JSON | - | -
auxiliary rate|The rate of auxiliaries across sentences|***auxiliary_rate***|Stanza(66)|Yes|No|String or JSON | - | -
conjunction rate|The rate of conjunctions across sentences|***conjunction_rate***|Stanza(66)|Yes|No|String or JSON | - | -
determiner rate|The rate of determiners across sentences|***determiner_rate***|Stanza(66)|Yes|No|String or JSON | - | -
interjection rate|The rate of interjections across sentences|***interjection_rate***|Stanza(66)|Yes|No|String or JSON | - | -
numeral rate|The rate of numerals across sentences|***numeral_rate***|Stanza(66)|Yes|No|String or JSON | - | -
particle rate|The rate of particles across sentences|***particle_rate***|Stanza(66)|Yes|No|String or JSON | - | -
pronoun rate|The rate of pronouns across sentences|***pronoun_rate***|Stanza(66)|Yes|No|String or JSON | - | -
proper noun rate|The rate of proper nouns across sentences|***proper_noun_rate***|Stanza(66)|Yes|No|String or JSON | - | -
punctuation rate|The rate of punctuations across sentences|***punctuation_rate***|Stanza(66)|Yes|No|String or JSON | - | -
subordinating conjunction rate|The rate of subordinating conjunctions across sentences|***subordinating_conjunction_rate***|Stanza(66)|Yes|No|String or JSON | - | -
symbol rate|The rate of symbols across sentences|***symbol_rate***|Stanza(66)|Yes|No|String or JSON | - | -
possessive rate|The rate of possessive words across sentences|***possessive_rate***|Stanza(66)|Yes|No|String or JSON | - | -
noun verb Ratio|The ratio of nouns to verbs across sentences|***noun_verb_ratio***|Stanza(66)|Yes|No|String or JSON | - | -
noun ratio|The ratio of nouns to the sum of nouns and verbns across sentences|***noun_ratio***|Stanza(66)|Yes|No|String or JSON | - | -
pronoun noun ratio|The ratio of pronouns to nouns across sentences|***pronoun_noun_ratio***|Stanza(66)|Yes|No|String or JSON | - | -
closed-class word rate |The proportions of determiners, pronouns, conjunctions and prepositions to all words across sentences|***closed_class_word_rate***|Stanza(66)|Yes|No|String or JSON | - | -
open-class word rate |The proportions of nouns, verbs, adjectives and adverbs to all words across sentences|***open_class_word_rate***|Stanza(66)|Yes|No|String or JSON | - | -
total dependency distance|The total distance of all dependencies across sentences|***total_dependency_distance***|Stanza(66)|Yes|No|String or JSON | - | -
average dependency distance|The average distance of all dependencies across sentences|***average_dependency_distance***|Stanza(66)|Yes|No|String or JSON | - | -
total dependencies|The total number of unique dependencies across sentences|***total_dependencies***|Stanza(66)|Yes|No|String or JSON | - | -
average dependencies|The average number of unique dependencies across sentences|***average_dependencies***|Stanza(66)|Yes|No|String or JSON | - | -
content density |The proportions of numebr of open class words to the numebr of close class words |***content_density***|Stanza(66)|Yes|No|String or JSON | - | -
idea density |he proportions of verbs, adjectives, adverbs, prepositions and conjucntions to all words across sentences |***idea_density***|Stanza(66)|Yes|No|String or JSON | - | -
honore's statistic |Calculated as R = (100*log(N))/(1-(V1)/(V)), where V is number of unique words, V1 is the number of words in the vocabulary only spoken once, and N is overall text length / number of words.  |***honore_statistic***|Stanza(66)|Yes|No|String or JSON | - | -
brunet's index |Calculated as N^(V^(-0.165)), where V is number of unique words and N is overall text length / number of words. Measure of lexical richness. Text-length insensitive version of TTR.  |***brunet_index***|Stanza(66)|Yes|No|String or JSON | - | -
type-token-ratio |The number of word types divided by the number of word tokens|***type_token_ratio***|Stanza(66)|Yes|No|String or JSON | - | -
word length |The mean length of words across the corpus|***word_length***|Stanza(66)|Yes|No|String or JSON | - | -
proportion of inflected verbs|The ratio of the number of inflected verbs to the number of verbs|***prop_inflected_verbs***|Stanza(66)|Yes|No|String or JSON | - | -
proportion of auxiliary verbs|The ratio of the number of auxiliary verbs to the number of verbs|***prop_auxiliary_verbs***|Stanza(66)|Yes|No|String or JSON | - | -
proportion of gerund verbs|The ratio of the number of gerund verbs to the number of verbs|***prop_gerund_verbs***|Stanza(66)|Yes|No|String or JSON | - | -
proportion of participles|The ratio of the number of particile verbs to the number of verbs|***prop_participles***|Stanza(66)|Yes|No|String or JSON | - | -
number of clauses|The number of clauses across the corpus|***num_clauses***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
clause rate|The number of clauses per sentence across the corpus|***clause_rate***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
number of dependent clauses|The number of Dependent Clauses |***num_dependent_clauses***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
dependent clauses rate|The rate of Dependent Clauses |***dependent_clause_rate***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
proportion of nouns with determiners|The proportion of nouns associated with a determiner|***prop_nouns_with_det***|Stanza(66)|Yes|No|String or JSON | - | -
proportion of nouns with adjectives|The proportion of nouns associated with a adjective|***prop_nouns_with_adj***|Stanza(66)|Yes|No|String or JSON | - | -
number of noun phrases|The number of Noun Phrases|***num_noun_phrases***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
noun phrase rate|The rate of Noun Phrases|***noun_phrase_rate***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
number of verb phrases|The number of Verb Phrases|***num_verb_phrases***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
verb phrase rate|The rate of Noun Phrases|***verb_phrase_rate***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
number of infinitive phrases|The number of Infinitive Phrases|***num_infinitive_phrases***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
infinitive phrase rate|The rate of Infinitive Phrases|***infinitive_phrase_rate***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
number of prepositional phrases|The number of Prepositional Phrases|***num_prepositional_phrases***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
prepositional phrase rate|The rate of Infinitive Phrases|***prepositional_phrase_rate***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
max yngve depth|The maximum Yngve Depth of each parse tree averaged over all sentences |***max_yngve_depth***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
mean yngve depth|The mean Yngve Depth of all nodes in a parse tree averaged over all sentences |***mean_yngve_depth***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
total yngve depth|The total Yngve Depth of all nodes in a parse tree averaged over all sentences |***total_yngve_depth***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
parse tree height|The height of parse tree averaged over all sentences |***parse_tree_height***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
number of discourse markers|The number of total discourse markers |***num_discourse_markers***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
discourse marker rate|The rate discourse markers across all sentences |***discourse_marker_rate***|CoreNLP(6)|Yes|Yes|String or JSON | - | -
