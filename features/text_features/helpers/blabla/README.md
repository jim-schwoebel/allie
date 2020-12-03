<p align="center">
<a href="https://novoic.com">
    <img src="https://assets.novoic.com/blabla.png" alt="blabla-logo" border="0">
</a>
  <br />
  <br />
<a href='https://novoic-blabla.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/novoic-blabla/badge/?version=latest' alt='Documentation Status' />
</a>
<a href='https://app.circleci.com/pipelines/github/novoic/blabla'>
    <img src='https://circleci.com/gh/novoic/blabla.svg?style=shield&circle-token=ee42afdb6e5cb9a4f34fec4fe31144d8bc9d1f99' alt='Build Status' />
</a>
</p>

_A Python package for clinical linguistic feature extraction in multiple languages_

For information about contributing, citing, licensing (including commercial licensing) and getting in touch, please see [our wiki](https://github.com/novoic/blabla/wiki).

Our documentation can be found [here](https://novoic-blabla.readthedocs.io/en/latest). Our paper can be found [here](https://arxiv.org/abs/2005.10219). For a list of features and their language support, see `FEATURES.md`.
  
## Setup
Note that BlaBla requires **Python version 3.6** or later. 

To install BlaBla from source:
```bash
git clone https://github.com/novoic/blabla.git
cd blabla
pip install .
```
To install BlaBla using PyPI:
```bash
pip install blabla
```


### Installing CoreNLP
For some features, BlaBla also requires [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) to be installed. See `FEATURES.md` for a list of these features.

To set up CoreNLP version 4.0.0, do `./setup_corenlp.sh` after changing `corenlp_dir` and `lang` if required. The legal `lang` parameters correspond to the languages available for CoreNLP:
* `english`
* `arabic`
* `chinese`
* `french`
* `german`
* `spanish`

After installation, or if you already have CoreNLP installed, let BlaBla know where to find it using `export CORENLP_HOME=/path/to/corenlp`.

CoreNLP also requires the [Java Developer Kit](https://www.oracle.com/java/technologies/javase-downloads.html) to be installed. To check whether it is already installed locally, run `$ javac -version`.

## Quickstart
Print the noun rate for some example text using Python (find the YAML configs inside the BlaBla repo):
```python
from blabla.document_processor import DocumentProcessor

with DocumentProcessor('stanza_config/stanza_config.yaml', 'en') as doc_proc:
    content = open('example_configs/example_document.txt').read()
    doc = doc_proc.analyze(content, 'string')

res = doc.compute_features(['noun_rate'])
print(res)
 ```  

Run BlaBla on a directory of text files and write the output to a csv file (find the YAML configs inside the BlaBla repo):
```bash
blabla compute-features -F example_configs/features.yaml -S stanza_config/stanza_config.yaml -i /path/to/text/files/dir -o blabla_features.csv -format string
```

For more details about usage, keep reading!

## Usage

BlaBla uses two config files for feature extraction. One of them specifies settings for the CoreNLP Server and the other specifies the list of features.

### Server config file
 
BlaBla comes with a predefined config file for Stanza and CoreNLP, which can be found at `stanza_config/stanza_config.yaml`. You don't need to modify any of these values. However, if want to run CoreNLP Server on a different port other than `9001`, change the port number.

### Input format

BlaBla supports two types of inputs. You can either send a free form text as a sentence or a paragraph or an array of JSONs. 

#### Free Text
You can process natural language represented in the form of free text with BlaBla. A sample text file is provided at `example_configs/example_document.txt`. Note that we specify the input format `"string"` when we call the `analyze` method. 

```python
from blabla.document_processor import DocumentProcessor

with DocumentProcessor('stanza_config/stanza_config.yaml', 'en') as doc_proc:
    content = open('example_configs/example_document.txt').read()
    doc = doc_proc.analyze(content, 'string')

res = doc.compute_features(['noun_rate', 'verb_rate'])
print(res)
 ```  
    
#### JSON Input
BlaBla requires word-level time stamps for phonetic features. The JSON format should be in a format that contains words and timestamps for each of the word in the text. Each JSON in the array corresponds to one sentence. A sample format is provided in the `example_configs/example_document.json` file in this repository. Note that we specify the input format `"json"` when we call the `analyze` method.
 
```python
from blabla.document_processor import DocumentProcessor

with DocumentProcessor('stanza_config/stanza_config.yaml', 'en') as doc_proc:
    content = open('example_configs/example_document.json').read()
    doc = doc_proc.analyze(content, 'json')
    
res = doc.compute_features(['speech_rate'])
print(res)
```
 
**Note:** Please make sure the compatibility between the feature, language and input format is maintained. If your input format is `string`, and you ask for a feature supported only in JSON format which requires timestamps (such as `total_pause_time`), the code will throw an exception. Refer to `FEATURES.md` file for more information.

### Command line interface

BlaBla can also be called using its command line interface (CLI). A sample command below shows how to use the CLI to process all the files in the a directory and dump the output features as a CSV file.

```bash
blabla compute-features -F example_configs/features.yaml -S stanza_config/stanza_config.yaml -i /path/to/text/files/dir -o blabla_features.csv -format string
```

When running the above CLI, you will need to provide the following arguments:
* `-F`: path to the `features.yaml` file defining the language and list of features.
* `-S`: path to the config file `stanza_config.yaml` containing the default server settings for Stanza and CoreNLP.
* `-i`: path to the input directory of text or JSON files.
* `-o`: path to the output CSV file. `test_output.csv` mentioned above.
* `-format`: the format of the input files (either `string` or `json`).
