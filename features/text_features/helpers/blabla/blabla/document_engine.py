import traceback

from blabla.sentence_aggregators.phonetic_and_phonological_feature_aggregator import (
    phonetic_and_phonological_feature_processor,
)
from blabla.sentence_aggregators.lexico_semantic_fearture_aggregator import (
    lexico_semantic_feature_processor,
)
from blabla.sentence_aggregators.morpho_syntactic_feature_aggregator import (
    morpho_syntactic_feature_processor,
)
from blabla.sentence_aggregators.syntactic_feature_aggregator import (
    syntactic_feature_processor,
)
from blabla.sentence_aggregators.discourse_and_pragmatic_feature_aggregator import (
    discourse_and_pragmatic_feature_processor,
)
import blabla.utils.settings as settings
from blabla.utils.exceptions import *



class Document(object):
    """This class represents the Document Engine class that defines all the features
    """

    def __init__(self, lang, nlp, client):
        self.lang = lang
        self.nlp = nlp
        self.client = client
        self.sentence_objs = []
        self.CONST_PT_SUPPORTED_LANGUAGES = ['ar', 'de', 'en', 'es', 'fr', 'zh-hant']

    def validate_features_list(self, feature_list):
        """Compute features

            Args:
                feature_list (str): A list of features to be extracted

            Returns:
                dict: A dictionary of features and their values
        """
        for feature in feature_list:
            if feature in settings.CONST_PT_FEATURES:
                if self.lang not in self.CONST_PT_SUPPORTED_LANGUAGES:
                    raise InvalidFeatureException(
                        'You have reqquested for a feature {} in language {} which are not compatible. Please check the FEATURED.md list'.format(
                            feature, self.lang
                        )
                    )

    def compute_features(self, feature_list, **kwargs):
        """Compute features

            Args:
                feature_list (list of str): A list of features to be extracted

            Returns:
                dict: A dictionary of features and their values
        """
        self.validate_features_list(feature_list)

        for sentence_obj in self.sentence_objs:
            sentence_obj.setup_dep_pt()
            if any(feature in settings.CONST_PT_FEATURES for feature in feature_list):
                sentence_obj.setup_const_pt()
                sentence_obj.setup_yngve_tree()

        features = {}
        for feature_name in feature_list:
            try:
                method_to_call = getattr(self, feature_name)
                result = method_to_call(**kwargs)
                features[feature_name] = result
            except AttributeError as e:
                raise InvalidFeatureException(
                    f'Please check the feature name. Feature name {feature_name} is invalid'
                )
        return features

    def _extract_phonetic_and_phonological_features(self, *features, **kwargs):
        """Extract phonetic and phonological features across all sentence objects

            Args:
                features (list): The list of features to be extracted
                kwargs (list): Optional arguments for threshold values

            Returns:
                dict: The dictionary containing the different features
        """
        features_dict = {}
        for feature in features:
            features_dict[feature] = phonetic_and_phonological_feature_processor(
                self.sentence_objs, feature, **kwargs
            )
        return features_dict

    def _extract_lexico_semantic_features(self, *features, **kwargs):
        """Extract lexico semantic features across all sentence objects

            Args:
                features (list): The list of features to be extracted
                kwargs (list): Optional arguments for threshold values

            Returns:
                dict: The dictionary containing the different features
        """
        features_dict = {}
        for feature in features:
            features_dict[feature] = lexico_semantic_feature_processor(
                self.sentence_objs, feature, **kwargs
            )
        return features_dict

    def _extract_morpho_syntactic_features(self, *features, **kwargs):
        """Extract morpho syntactic features across all sentence objects

            Args:
                features (list): The list of features to be extracted
                kwargs (list): Optional arguments for threshold values

            Returns:
                dict: The dictionary containing the different features
        """
        features_dict = {}
        for feature in features:
            features_dict[feature] = morpho_syntactic_feature_processor(
                self.sentence_objs, feature, **kwargs
            )
        return features_dict

    def _extract_syntactic_features(self, *features, **kwargs):
        """Extract syntactic features across all sentence objects

            Args:
                features (list): The list of features to be extracted
                kwargs (list): Optional arguments for threshold values

            Returns:
                dict: The dictionary containing the different features
        """
        features_dict = {}
        for feature in features:
            features_dict[feature] = syntactic_feature_processor(
                self.sentence_objs, feature, **kwargs
            )
        return features_dict

    def _extract_discourse_and_pragmatic_feature_processor(self, *features, **kwargs):
        """Extract discourse and pragmatic features across all sentence objects

            Args:
                features (list): The list of features to be extracted
                kwargs (list): Optional arguments for threshold values

            Returns:
                dict: The dictionary containing the different features
        """
        features_dict = {}
        for feature in features:
            features_dict[feature] = discourse_and_pragmatic_feature_processor(
                self.sentence_objs, feature, **kwargs
            )
        return features_dict

    def num_pauses(self, **kwargs):
        """Method to extract the number of pauses

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                int: The number of pauses
        """
        return self._extract_phonetic_and_phonological_features('num_pauses', **kwargs)[
            'num_pauses'
        ]

    def total_pause_time(self, **kwargs):
        """Method to extract the total pause time

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The total pause time
        """
        return self._extract_phonetic_and_phonological_features(
            'total_pause_time', **kwargs
        )['total_pause_time']

    def mean_pause_duration(self, **kwargs):
        """Method to extract the mean pause duration

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The mean pause duration
        """
        return self._extract_phonetic_and_phonological_features(
            'mean_pause_duration', **kwargs
        )['mean_pause_duration']

    def between_utterance_pause_duration(self, **kwargs):
        """Method to extract the between utterance pause duration

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The between utterance pause duration
        """
        return self._extract_phonetic_and_phonological_features(
            'between_utterance_pause_duration', **kwargs
        )['between_utterance_pause_duration']

    def hesitation_ratio(self, **kwargs):
        """Method to extract the hesitation ratio

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The hesitation ratio
        """
        return self._extract_phonetic_and_phonological_features(
            'hesitation_ratio', **kwargs
        )['hesitation_ratio']

    def speech_rate(self, **kwargs):
        """Method to extract the speech rate

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The speech rate
        """
        return self._extract_phonetic_and_phonological_features('speech_rate', **kwargs)[
            'speech_rate'
        ]

    def maximum_speech_rate(self, **kwargs):
        """Method to extract the maximum speech rate

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The maximum speech rate
        """
        return self._extract_phonetic_and_phonological_features(
            'maximum_speech_rate', **kwargs
        )['maximum_speech_rate']

    def total_phonation_time(self, **kwargs):
        """Method to extract the total phonation time

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The total phonation time
        """
        return self._extract_phonetic_and_phonological_features(
            'total_phonation_time', **kwargs
        )['total_phonation_time']

    def std_phonation_time(self, **kwargs):
        """Method to extract the standardized phonation time

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The standardized phonation time
        """
        return self._extract_phonetic_and_phonological_features(
            'standardized_phonation_time', **kwargs
        )['standardized_phonation_time']

    def total_locution_time(self, **kwargs):
        """Method to extract the total locution time

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The total locution time
        """
        return self._extract_phonetic_and_phonological_features(
            'total_locution_time', **kwargs
        )['total_locution_time']

    def adjective_rate(self, **kwargs):
        """Extract the adjective rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The adjective rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('adjective_rate', **kwargs)['adjective_rate']

    def adposition_rate(self, **kwargs):
        """Extract the adposition rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The adposition rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('adposition_rate', **kwargs)['adposition_rate']

    def adverb_rate(self, **kwargs):
        """Extract the adverb rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The adverb rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('adverb_rate', **kwargs)['adverb_rate']

    def auxiliary_rate(self, **kwargs):
        """Extract the auxiliary rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The auxiliary rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('auxiliary_rate', **kwargs)['auxiliary_rate']

    def determiner_rate(self, **kwargs):
        """Extract the determiner rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The determiner rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('determiner_rate', **kwargs)['determiner_rate']

    def interjection_rate(self, **kwargs):
        """Extract the interjection rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The interjection rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('interjection_rate', **kwargs)['interjection_rate']

    def noun_rate(self, **kwargs):
        """Extract the noun rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The noun rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('noun_rate', **kwargs)['noun_rate']

    def numeral_rate(self, **kwargs):
        """Extract the numeral rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The numeral rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('numeral_rate', **kwargs)['numeral_rate']

    def particle_rate(self, **kwargs):
        """Extract the particle rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The particle rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('particle_rate', **kwargs)['particle_rate']

    def pronoun_rate(self, **kwargs):
        """Extract the pronoun rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The pronoun rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('pronoun_rate', **kwargs)['pronoun_rate']

    def proper_noun_rate(self, **kwargs):
        """Extract the proper_noun rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The proper_noun rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('proper_noun_rate', **kwargs)['proper_noun_rate']

    def punctuation_rate(self, **kwargs):
        """Extract the punctuation rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The punctuation rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('punctuation_rate', **kwargs)['punctuation_rate']

    def subordinating_conjunction_rate(self, **kwargs):
        """Extract the subordinating_conjunction rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The subordinating_conjunction rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('subordinating_conjunction_rate', **kwargs)['subordinating_conjunction_rate']

    def symbol_rate(self, **kwargs):
        """Extract the symbol rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The symbol rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('symbol_rate', **kwargs)['symbol_rate']

    def verb_rate(self, **kwargs):
        """Extract the verb rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                flaot: The verb rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('verb_rate', **kwargs)['verb_rate']

    def demonstrative_rate(self, **kwargs):
        """Extract the demonstrative rate

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The demonstrative rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('demonstrative_rate', **kwargs)[
            'demonstrative_rate'
        ]

    def conjunction_rate(self, **kwargs):
        """Extract the conjunction rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The conjunction rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('conjunction_rate', **kwargs)[
            'conjunction_rate'
        ]

    def possessive_rate(self, **kwargs):
        """Extract the possesive rate.
            Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3642700/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The possesive rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('possessive_rate', **kwargs)[
            'possessive_rate'
        ]

    def noun_verb_ratio(self, **kwargs):
        """Extract the noun to verb rate.
            Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The noun to verb rate across all sentence objects
        """
        return self._extract_lexico_semantic_features('noun_verb_ratio', **kwargs)[
            'noun_verb_ratio'
        ]

    def noun_ratio(self, **kwargs):
        """Extract the noun ratio.
            Ref:https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The noun ratio across all sentence objects
        """
        return self._extract_lexico_semantic_features('noun_ratio', **kwargs)[
            'noun_ratio'
        ]

    def pronoun_noun_ratio(self, **kwargs):
        """Extract the pronoun to noun ratio.
            Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The pronoun to noun ratio across all sentence objects
        """
        return self._extract_lexico_semantic_features('pronoun_noun_ratio', **kwargs)[
            'pronoun_noun_ratio'
        ]

    def total_dependency_distance(self, **kwargs):
        """Extract the total dependency distance.
            Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The total dependency distance across all sentence objects
        """
        return self._extract_lexico_semantic_features('total_dependency_distance', **kwargs)[
            'total_dependency_distance'
        ]

    def average_dependency_distance(self, **kwargs):
        """Extract the average dependency distance.
            Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The average dependency distance across all sentence objects
        """
        return self._extract_lexico_semantic_features('average_dependency_distance', **kwargs)[
            'average_dependency_distance'
        ]

    def total_dependencies(self, **kwargs):
        """Extract the number of unique dependency relations.
            Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The total number of unique dependencies across all sentence objects
        """
        return self._extract_lexico_semantic_features('total_dependencies', **kwargs)[
            'total_dependencies'
        ]

    def average_dependencies(self, **kwargs):
        """Extract the average number of unique dependency relations.
            Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The average number of unique dependencies across all sentence objects
        """
        return self._extract_lexico_semantic_features('average_dependencies', **kwargs)[
            'average_dependencies'
        ]

    def closed_class_word_rate(self, **kwargs):
        """Extract the proportion of closed class words.
            Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The proportion of closed class words across all sentence objects
        """
        return self._extract_lexico_semantic_features(
            'closed_class_word_rate', **kwargs
        )['closed_class_word_rate']

    def open_class_word_rate(self, **kwargs):
        """Extract the prooportion of open class words.
            Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The prooportion of open class words across all sentence objects
        """
        return self._extract_lexico_semantic_features('open_class_word_rate', **kwargs)[
            'open_class_word_rate'
        ]

    def content_density(self, **kwargs):
        """Extract the content density
            Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The content density across all sentence objects
        """
        return self._extract_lexico_semantic_features('content_density', **kwargs)[
            'content_density'
        ]

    def idea_density(self, **kwargs):
        """Extract the idea density
            Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The idea density across all sentence objects
        """
        return self._extract_lexico_semantic_features('idea_density', **kwargs)[
            'idea_density'
        ]

    def honore_statistic(self, **kwargs):
        """Extract the honore statistic
            Ref: https://www.aclweb.org/anthology/W16-1902.pdf

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The honore statistic across all sentence objects
        """
        return self._extract_lexico_semantic_features('honore_statistic', **kwargs)[
            'honore_statistic'
        ]

    def brunet_index(self, **kwargs):
        """Extract the brunet's index.
            Ref: https://www.aclweb.org/anthology/W16-1902.pdf

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The brunet's index across all sentence objects
        """
        return self._extract_lexico_semantic_features('brunet_index', **kwargs)[
            'brunet_index'
        ]

    def type_token_ratio(self, **kwargs):
        """Extract the type to token ratio.
            Ref https://www.tandfonline.com/doi/abs/10.1080/02687038.2017.1303441

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The type to token ratio across all sentence objects
        """
        return self._extract_lexico_semantic_features('type_token_ratio', **kwargs)[
            'type_token_ratio'
        ]

    def word_length(self, **kwargs):
        """Extract the mean word length.
            Ref: https://pubmed.ncbi.nlm.nih.gov/26484921/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The mean word length all sentence objects
        """
        return self._extract_lexico_semantic_features('word_length', **kwargs)[
            'word_length'
        ]

    def prop_inflected_verbs(self, **kwargs):
        """Extract proportion of inflected verbs.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The proportion of inflected verbs across all sentence objects
        """
        return self._extract_morpho_syntactic_features("prop_inflected_verbs", **kwargs)[
            "prop_inflected_verbs"
        ]

    def prop_auxiliary_verbs(self, **kwargs):
        """Extract the proportion of auxiliary verbs
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The proportion of auxiliary verbs across all sentence objects
        """
        return self._extract_morpho_syntactic_features('prop_auxiliary_verbs', **kwargs)[
            'prop_auxiliary_verbs'
        ]

    def prop_gerund_verbs(self, **kwargs):
        """Extract the proportion of gerund verbs.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The the proportion of gerund verbs across all sentence objects
        """
        return self._extract_morpho_syntactic_features('prop_gerund_verbs', **kwargs)[
            'prop_gerund_verbs'
        ]

    def prop_participles(self, **kwargs):
        """Extract the proportion of participle verbs.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
               float:  The proportion of participle verbs across all sentence objects
        """
        return self._extract_morpho_syntactic_features('prop_participles', **kwargs)[
            'prop_participles'
        ]

    def num_noun_phrases(self, **kwargs):
        """Extract the number of noun phrases.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The number of noun phrases across all sentence objects
        """
        return self._extract_syntactic_features('num_noun_phrases', **kwargs)[
            'num_noun_phrases'
        ]

    def noun_phrase_rate(self, **kwargs):
        """Extract the number of noun phrases.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
               float: The noun phrase rate across all sentence objects
        """
        return self._extract_syntactic_features('noun_phrase_rate', **kwargs)[
            'noun_phrase_rate'
        ]

    def num_verb_phrases(self, **kwargs):
        """Extract the number of verb phrases.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
               float: The number of verb phrases across all sentence objects
        """
        return self._extract_syntactic_features('num_verb_phrases')['num_verb_phrases']

    def verb_phrase_rate(self, **kwargs):
        """Extract the number of noun phrases.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The noun phrase rate across all sentence objects
        """
        return self._extract_syntactic_features('verb_phrase_rate', **kwargs)[
            'verb_phrase_rate'
        ]

    def num_prepositional_phrases(self, **kwargs):
        """Extract the number of prepositional phrases.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
               float:  The number of prepositional phrases across all sentence objects
        """
        return self._extract_syntactic_features('num_prepositional_phrases', **kwargs)[
            'num_prepositional_phrases'
        ]

    def prepositional_phrase_rate(self, **kwargs):
        """Extract the number of prepositional phrases.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The prepositional phrases rate across all sentence objects
        """
        return self._extract_syntactic_features('prepositional_phrase_rate', **kwargs)[
            'prepositional_phrase_rate'
        ]

    def num_clauses(self, **kwargs):
        """Extract the number of clauses.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The number of clauses across all sentence objects
        """
        return self._extract_syntactic_features('num_clauses', **kwargs)['num_clauses']

    def clause_rate(self, **kwargs):
        """Extract the number of clauses.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The number of clauses across all sentence objects
        """
        return self._extract_syntactic_features('clause_rate', **kwargs)['clause_rate']

    def num_infinitive_phrases(self, **kwargs):
        """Extract the number of infinitive phrases.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The number of infinitive phrases across all sentence objects
        """
        return self._extract_syntactic_features('num_infinitive_phrases', **kwargs)[
            'num_infinitive_phrases'
        ]

    def infinitive_phrase_rate(self, **kwargs):
        """Extract the number of infinitive phrases.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The infinitive phrases rate across all sentence objects
        """
        return self._extract_syntactic_features('infinitive_phrase_rate', **kwargs)[
            'infinitive_phrase_rate'
        ]

    def num_dependent_clauses(self, **kwargs):
        """Extract the number of dependent clauses.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/
            Args:
                kwargs (list): Optional arguments for threshold values
            Returns:
                float: The number of dependent clauses across all sentence objects
        """
        return self._extract_syntactic_features('num_dependent_clauses', **kwargs)[
            'num_dependent_clauses'
        ]

    def dependent_clause_rate(self, **kwargs):
        """Extract the dependent clauses rate.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                The dependent clauses rate across all sentence objects
        """
        return self._extract_syntactic_features('dependent_clause_rate', **kwargs)[
            'dependent_clause_rate'
        ]

    def prop_nouns_with_det(self, **kwargs):
        """Extract the proportion of nouns with determiners.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The proportion of nouns with determiners across all sentence objects
        """
        return self._extract_syntactic_features('prop_nouns_with_det', **kwargs)[
            'prop_nouns_with_det'
        ]

    def prop_nouns_with_adj(self, **kwargs):
        """Extract the proportion of nouns with determiners.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The proportion of nouns with determiners across all sentence objects
        """
        return self._extract_syntactic_features('prop_nouns_with_adj', **kwargs)[
            'prop_nouns_with_adj'
        ]

    def max_yngve_depth(self, **kwargs):
        """Extract the maximum yngve depth.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The maximum yngve depth across all sentence objects
        """
        return self._extract_syntactic_features('max_yngve_depth', **kwargs)[
            'max_yngve_depth'
        ]

    def mean_yngve_depth(self, **kwargs):
        """Extract the mean yngve depth.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The mean yngve depth across all sentence objects
        """
        return self._extract_syntactic_features('mean_yngve_depth', **kwargs)[
            'mean_yngve_depth'
        ]

    def total_yngve_depth(self, **kwargs):
        """Extract the total yngve depth.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The total yngve depth across all sentence objects
        """
        return self._extract_syntactic_features('total_yngve_depth', **kwargs)[
            'total_yngve_depth'
        ]

    def parse_tree_height(self, **kwargs):
        """Extract the constituency parse tree height.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The constituency parse tree height across all sentence objects
        """
        return self._extract_syntactic_features('parse_tree_height', **kwargs)[
            'parse_tree_height'
        ]

    def num_discourse_markers(self, **kwargs):
        """Extract the number of discourse markers.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The number of discourse markers across all sentence objects
        """
        return self._extract_discourse_and_pragmatic_feature_processor(
            'num_discourse_markers', **kwargs
        )['num_discourse_markers']

    def discourse_marker_rate(self, **kwargs):
        """Extract the number of discourse markers.
            Ref: https://pubmed.ncbi.nlm.nih.gov/28321196/

            Args:
                kwargs (list): Optional arguments for threshold values

            Returns:
                float: The discourse markers rate across all sentence objects
        """
        return self._extract_discourse_and_pragmatic_feature_processor(
            'discourse_marker_rate', **kwargs
        )['discourse_marker_rate']
