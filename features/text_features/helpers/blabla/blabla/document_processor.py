import stanza
import yaml
import os
import json
import blabla.utils.global_params as global_params
from os import environ
from stanza.server import CoreNLPClient
from blabla.sentence_processor.sentence_processing_engine import Sentence
from blabla.document_engine import Document
from blabla.utils.exceptions import *
import blabla.utils.global_params as global_params
import blabla.utils.settings as settings


class DocumentProcessor(object):
    """This class represents the Document Processor class that processes the whole input document
	"""

    def __init__(self, config_path, lang):
        self.config = yaml.load(open(config_path, "r"))
        self.client = None
        self.lang = lang

    def __enter__(self):
        if environ.get("CORENLP_HOME") is None:
            raise EnvPathException(
                "The CORENLP_HOME path was not found. Please export it pointing to the directory that contains the CoreNLP resources"
            )
        my_path = os.path.abspath(os.path.dirname(__file__))
        settings.init()
        settings.LANGUAGE = self.lang
        stanza.download(self.lang, dir=self.config["stanza"]["dir"])
        self.nlp = stanza.Pipeline(**self.config["stanza"], lang=self.lang)
        language_properties_fp = os.path.join(
            my_path, "language_resources", self.lang + "_properties.txt"
        )
        self.client = CoreNLPClient(
            properties=language_properties_fp, **self.config["corenlp"]
        )
        return self

    def break_json_into_chunks(self, doc_json):
        """Convert an input json to a list of sentences

			Args:
				doc_json (dict): The input json representing the input document

			Returns:
				list : The list of sentences with raw text
				list: The list of sentences as jsons
		"""
        raw_sentences = []
        sentence_jsons = []
        try:
            for sent_json in doc_json:
                sentence_jsons.append(sent_json)
                sent_text = " ".join([word["word"] for word in sent_json["words"]])
                raw_sentences.append(sent_text)
        except Exception as e:
            raise InavlidJSONFileException(
                "The input JSON file you provided could not be analysed. Please check the example format provided"
            )
        return raw_sentences, sentence_jsons

    def break_text_into_sentences(self, text, force_split):
        """Break the input raw text string into sentences using Stanza

			Args:
				doc_json (dict): The input json representing the input document
                force_split (bool): If True, split sentences on newline, else use Stanza tokenization

			Returns:
				list : The list of sentences with raw text
				list : The list of sentences as jsons
		"""
        sentences = []
        if force_split:
            sentences = [s for s in text.split('\n') if s]
        else:
            stanza_doc = self.nlp(text)
            for sentence in stanza_doc.sentences:
                sentences.append(sentence.text)
        return sentences

    def analyze(self, doc, input_format, force_split=False):
        """Method to analyze the input as either a json or a string and return back a Document object

			Args:
				doc (json / string): The input that needs to be analyzed using Stanza

			Returns:
				Document: The Document object
		"""
        if input_format.lower() not in ["string", "json"]:
            raise InavlidFormatException(
                "Please provide the format as either 'string' or 'json'"
            )

        settings.INPUT_FORMAT = input_format.lower()
        doc_obj = Document(self.lang, self.nlp, self.client)
        if settings.INPUT_FORMAT == "json":  # the input format here is json
            doc = json.loads(doc)
            raw_sentences, sentence_jsons = self.break_json_into_chunks(doc)
            for raw_sent, sent_json in zip(raw_sentences, sentence_jsons):
                sentence = Sentence(
                    self.lang, self.nlp, self.client, raw_sent, sent_json
                )
                sentence.json = sent_json
                doc_obj.sentence_objs.append(sentence)
        else:  # the input format here is string
            raw_sentences = self.break_text_into_sentences(doc, force_split)
            for raw_sent in raw_sentences:
                sentence = Sentence(self.lang, self.nlp, self.client, raw_sent)
                doc_obj.sentence_objs.append(sentence)
        return doc_obj

    def __exit__(self, exc_type, exc_value, tb):
        """ Method to stop the CoreNLP client"""
        if self.client is not None:
            self.client.stop()
