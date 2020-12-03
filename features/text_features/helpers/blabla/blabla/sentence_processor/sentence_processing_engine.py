from nltk.tree import Tree
from blabla.sentence_processor.yngve_tree import YngveNode
from blabla.sentence_processor.pos_tag_counting_engine import PosTagCounter
from blabla.utils.exceptions import *
import blabla.utils.settings as settings
from blabla.utils.global_params import *
import os
import stanza
import nltk
import numpy as np
import math
import json

## Ref: http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
class Sentence(object):
	"""The class that is responsible for processing a sentence and extracting the fundamental blocks such as
		the dependency parse tree, constituency parse tree and other low level features for aggregation later.
	"""

	def __init__(self, language, nlp, client, raw_text, sent_json=None):
		"""The initialization method for the Sentence class

		Args:
			language (str): The language of the raw_text
			nlp (obj): The Stanza NLP object.
			client (obj): The CoreNLP client object.
			raw_text (str): The raw text 
			sent_json (JSON): The json string of a sentence if the input is a JSON
		Returns:
			None:
		"""
		self.lang = language
		self._pos_tag_counter = None
		self._sent = raw_text
		self._stanza_doc = None
		self._const_pt = None
		self._yngve_tree_root = None
		self.nlp = nlp
		self.client = client
		self.json = sent_json

	@property
	def pos_tag_counter(self):
		"""A property method to return the part of speech counter object"""
		return self._pos_tag_counter

	@property
	def sent(self):
		"""A property method to return the raw text of the current sentence object"""
		return self._sent

	@property
	def stanza_doc(self):
		"""A property method to return the Stanza object"""
		return self._stanza_doc

	@property
	def yngve_tree_root(self):
		"""A property method to return the root node of the Yngve tree"""
		return self._yngve_tree_root

	@property
	def const_pt(self):
		"""A property method to return the constituency parse tree"""
		return self._const_pt

	@property
	def tot_num_characters(self):
		"""A property method to return the total number of characters in the current sentence"""
		return math.sum([len(word) for word in self._stanza_doc.sentences[0].words])

	@property
	def speech_time(self):
		"""A property method to return the speech time"""
		return math.fsum([word['duration'] for word in self.json['words']])

	@property
	def start_time(self):
		"""A property method to return the start time of the current sentence"""
		return self.json['words'][0]['start_time']

	@property
	def end_time(self):
		"""A property method to return the end time of the current sentence"""
		return self.json['words'][-1]['start_time'] + self.json['words'][-1]['duration']

	@property
	def locution_time(self):
		"""A property method to return the locution time of the current sentence"""
		start_time = self.start_time
		end_time = self.end_time
		return end_time - start_time

	@property
	def words_per_min(self):
		"""A property method to return the words per min for the current sentence"""
		return (self.num_words() / self.locution_time()) * 60.0

	def _get_gaps(self):
		gaps = []
		for prev_word, next_word in zip(self.json['words'][:-1], self.json['words'][1:]):
			prev_end_time = prev_word['start_time'] + prev_word['duration']
			gaps.append(next_word['start_time'] - prev_end_time)
		return gaps

	def num_pauses(self, pause_duration):
		"""A method to calculate the number of pauses in the current sentence
			Args:
				pause_duration (float): The value for the duration of the pause
			Returns:
				int: The number of pauses
		"""
		gaps = self._get_gaps()     
		return sum(gap > pause_duration for gap in gaps)

	def tot_pause_time(self, pause_duration):
		"""A method to calculate the total pause time in the current sentence
			Args:
				pause_duration (float): The value for the duration of the pause
			Returns:
				float: The total pause time of the sentence
		"""
		return sum([gap for gap in self._get_gaps()  if gap > pause_duration])

	def analyze_text(self):
		"""A method to get the Stanza document
			Args:
				None: 
			Returns:
				doc (Stanza): The Stanza document object
		"""
		doc = self.nlp(self._sent)
		return doc

	def const_parse_tree(self):
		"""A method to get the constituency parse tree
			Args:
				None: 
			Returns:
				parseTree (CoreNLP): The CoreNLP constituency parse tree object
		"""
		document = self.client.annotate(self._sent)
		document = json.loads(document.text)
		pt = document["sentences"][0]["parse"]
		parseTree = Tree.fromstring(pt)
		return parseTree

	def num_words(self):
		"""A method to get the number of words
			Args:
				None: 
			Returns:
				int: The number of words in the current sentence
		"""
		return len(self._stanza_doc.sentences[0].words)

	def _navigate_and_score_leaves(self, yngve_node, score_so_far):
		"""A method to assign Yngve scores to the leaf nodes
			Args:
				None: 
			Returns:
				Yngve_Tree: The Yngve tree with the scores for the leaf nodes
		"""
		if len(yngve_node.children) == 0:
			yngve_node.score = score_so_far
		else:  # it has child nodes
			for child in yngve_node.children:
				self._navigate_and_score_leaves(child, score_so_far + child.score)

	def _traverse_and_build_yngve_tree(self, start_node, parent_node):
		"""A method to construct the Yngve tree
			Args:
				None: 
			Returns:
				Yngve_Tree: The Yngve tree constructed from the constituency parse tree
		"""
		score = 0
		for child in start_node[::-1]:
			if isinstance(child, str):
				curr_node = YngveNode(child, 0, parent_node)
			elif isinstance(child, nltk.tree.Tree):
				curr_node = YngveNode(child.label(), score, parent_node)
				score += 1
				self._traverse_and_build_yngve_tree(child, curr_node)

	def yngve_tree(self):
		"""The main method to construct the Yngve tree and assign values to all leaf nodes
			Args:
				None
			Returns:
				yngve_tree_root_node (YngveNode): The root node of the yngve tree
		"""
		sent_child = self._const_pt[0]
		yngve_tree_root_node = YngveNode("S", 0)
		self._traverse_and_build_yngve_tree(sent_child, yngve_tree_root_node)
		self._navigate_and_score_leaves(
			yngve_tree_root_node, yngve_tree_root_node.score
		)
		return yngve_tree_root_node

	def setup_dep_pt(self):
		"""A method to construct the dependency parse tree
			Args:
				None
			Returns:
				None:
		"""
		if len(self._sent) == 0:
			raise EmptyStringException('The input string is empty')

		try:
			self._stanza_doc = self.analyze_text()
		except Exception as e:
			raise DependencyParsingTreeException('Dependency parse tree set up failed')

		try:
			self._pos_tag_counter = PosTagCounter(self._stanza_doc)
		except Exception as e:
			raise POSTagExtractionFailedException('POS Tag counter failed')

	def setup_const_pt(self):
		"""A method to construct the constituency parse tree
			Args:
				None
			Returns:
				None:
		"""
		try:
			self._const_pt = self.const_parse_tree()
		except Exception as e:
			raise ConstituencyTreeParsingException(
				'Constituency parse tree set up failed. Please check if the input format (json/string) is mentioned correctly'
			)

	def setup_yngve_tree(self):
		"""A method to construct the Yngve tree
			Args:
				None
			Returns:
				None:
		"""
		try:
			self._yngve_tree_root = self.yngve_tree()
		except Exception as e:
			raise YngveTreeConstructionException('Yngve tree set up failed')
