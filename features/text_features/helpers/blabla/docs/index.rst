:github_url: https://github.com/novoic/surfboard

BlaBla: Linguistic Feature Extraction for Clinical Analysis in Multiple Languages 
=================================================================================

.. toctree::
    :maxdepth: 2

    intro_installation


BlaBla
-------------------------------------------

At the heart of BlaBla is the :code:`DocumentProcessor` and the :code:`Document` class. You have to import the :code:`DocumentProcessor` class to process a piece of input text as shown in the below piece of code.

.. code-block:: python

    from bla_bla.document_processor import DocumentProcessor
    with DocumentProcessor("stanza_config/stanza_config.yaml", "en") as doc_proc:
        content = "The picture shows a boy walking to the kitchen to pick a cookie from the cookie jar."
        doc = doc_proc.analyze(content, "string")
        res_json = doc.compute_features("noun_rate")
        print(res_json)

Under the hood, the :code:`DocumentProcessor` object has an :code:`analyze` method that will return an object of type :code:`Document` class which can be used to compute features

.. toctree::
    :maxdepth: 2

    document_processor

.. toctree::
    :maxdepth: 2

    document_engine

Features Table
==================

* :ref:`genindex`
* :ref:`search`
