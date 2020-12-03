#!/bin/bash
set -e

lang=english
corenlp_dir="$HOME/corenlp"

if [ -d $corenlp_dir ]; then
    echo "Error: directory $corenlp_dir already exists"
    exit 1
fi

curl https://nlp.stanford.edu/software/stanford-corenlp-4.0.0.zip -o stanford-corenlp-4.0.0.zip
unzip -qq stanford-corenlp-4.0.0.zip
mv stanford-corenlp-4.0.0 $corenlp_dir

curl https://nlp.stanford.edu/software/stanford-corenlp-4.0.0-models-$lang.jar -o $corenlp_dir/stanford-corenlp-4.0.0-models-$lang.jar

echo "CoreNLP ($lang) successfully installed at $corenlp_dir"
echo "Now and in the future, run 'export CORENLP_HOME=$corenlp_dir' before using BlaBla or add this command to your .bashrc/.profile or equivalent file"
