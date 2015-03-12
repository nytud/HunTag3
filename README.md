HunTag3 - A sequential tagger for NLP combining the Scikit-learn/LinearRegressionClassifier linear classifier and Hidden Markov Models.

Based on training data, HunTag3 can perform any kind of sequential sentence
tagging and has been used for NP chunking and Named Entity Recognition.

HungTag3 is a fork of the HunTag (https://github.com/recski/HunTag) project.

# Differences between HunTag3 and HunTag
HunTag has numerous features that have been replaced in multiple steps (see commits). These are:
- Latin-2 encoding -> UTF-8 encoding
- Python2 -> Python3
- cType -> NumPy/SciPy arrays
- Liblinear -> Scikit-learn/LinearRegressionClassifier
- Performance: memory comnsumption is 25% lower, training time is 22% higher 
  (measured on the Szeged TreeBank NP chunker task)

Some transitional versions also exist, but they are not supported. In this repository the following transitional versions (commits) can be found:

- Code cleanup: Latin-2 + Python2 + cType + Liblinear
- Convert to Python3-UTF-8: UTF-8 + Python3 + cType + Liblinear
- Convert to Scikit-learn-NumPy/SciPy: UTF-8 + Python3 + NumPy/SciPy + Scikit-learn
- Add features (Stable): This version introduces numerous extra features over the original HunTag

# Requirements

- For the Python3 + Liblinear version: Liblinear=1.94, no additional patch required
- For later versions: NumPy, SciPy, Scikit-learn
- Optional: CRFsuite

# Data format

- Input data must be a tab-separated file with one word per line
- An empty line to mark sentence boundaries
- Each line must contain the same number of fields
- Conventionally the last field must contain the correct label for the word (it is possible to use other fields for the label, but it must be set using the apropriate command line arguments).
    - The label may be in the BI format used at CoNLL shared tasks (e. g. B-NP to mark the first word of a noun phrase, I-NP to mark the rest and O to mark words outside an NP)
    - Or in the so-called BIE1 format which has a seperate symbol for words constituting a chunk themselves (1-NP) and one for the last words of multi-word phrases (E-NP)
    - The first two characters of labels should always conform to one of these two conventions, the rest may be any string describing the category

# Features

The flexibility of Huntag comes from the fact that it will generate any kind of features from the input data given the **appropriate python functions** (please refer to features.py and the config files).
Several dozens of features used regularly in NLP tasks are already implemented in the file features.py, however the user is encouraged to add any number of her own.

Once the desired features are implemented, a data set and a configuration file containing the list of feature functions to be used are all HunTag needs to perform training and tagging.

# Config file
The configuration file lists the features that are to be used for a given task. The feature file may start with a command specifying the default radius for features. This is non-mandatory. Example:

    !defaultRadius 5

Next, it can give values to variables that shall be used by the featurizing methods.
For example, the following three lines set the parameters of the feature called *krpatt*

    let krpatt minLength 2
    let krpatt maxLength 99
    let krpatt lang hu

The second field specifies the name of the feature, the third a key, the fourth a numeric value. The dictionary of key-value pairs will be passed to the feature.

After this come the actual assignments of feature names to features. Examples:

    token ngr ngrams 0
    sentence bwsamecases isBetweenSameCases 1
    lex street hunner/lex/streetname.lex 0
    token lemmalowered lemmaLowered 0,2

The first keyword can have three values, token, lex and sentence. For example, in the first example line above, the feature name ngr will be assigned to the python function ngrams() that returns a feature string for the given token. The third argument is a column or comma-separated list of columns. It specifies which fields of the input should be passed to the feature function. Counting starts from zero.

For sentence features, the input is aggregated sentence-wise into a list, and this list is then passed to the feature function. This function should return a list consisting of one feature string for each of the tokens of the sentence.

For lex features, the second argument specifies a lexicon file rather than a python function name. The specified token field is matched against this lexicon file.


# Usage
HunTag may be run in any of the following modes (see startHuntag.sh for overview and *huntag.py --help* for details):

## train
Used to train a model given a training corpus and a set of feature functions. When run in TRAIN mode, HunTag creates three files, one containing the model and two listing features and labels and the integers they are mapped to when passed to the learner. With the --model option set to NAME, the three files will be stored under NAME.model, NAME.featureNumbers and NAME.labelNumbers respectively.

    cat TRAINING_DATA | python3 huntag.py train OPTIONS

Mandatory options:
- -c FILE, --config-file=FILE
    - read feature configuration from FILE
- -m NAME, --model=NAME
    - name of model and lists

Non-mandatory options:
- -f FILE, --feature-file=FILE
    - write training events to FILE

## bigram-train
Used to train a bigram language model using a given field of the training data

    cat TRAINING_DATA | python3 huntag.py bigram-train OPTIONS

Mandatory options:
- -m NAME, --model=NAME
    - name of model file and lists

## tag
Used to tag input. Given a maxent model providing the value P(l|w) for all labels l and words (set of feature values) w, and a bigram language model supplying P(l|l0) for all pairs of labels, HunTag will assign to each sentence the most likely label sequence.

    cat INPUT | python3 huntag.py tag OPTIONS

Mandatory options:
- -m NAME, --model=NAME
    - name of model file and lists
- -c FILE, --config-file=FILE
    - read feature configuration from FILE

Non-mandatory options:
- -l L, --language-model-weight=L
    - set weight of the language model to L (default is 1)

## most-informative-features
Generates a feature ranking by counting label probabilities (for each label) and  frequency per feature and sort them in decreasing order of confidence and frequency. This output is usefull for inspecting features quality.

    cat TRAINING_DATA | python3 huntag.py most-informative-features OPTIONS > modelName.mostInformativeFeatures

Mandatory options:
- -m NAME, --model=NAME
    - name of model file and lists
- -c FILE, --config-file=FILE
    - read feature configuration from FILE

## tag --printWeights N
Usefull for inspecting feature weights (per label) assigned by the MaxEnt learner. (As name suggests, training must happen before tagging.)
Negative weights mean negative correlation, which is also usefull.

    python3 huntag.py tag --printWeights N OPTIONS > modelName.modelWeights

Mandatory options:
- N is the number of features to print (default: 100)
- -m NAME, --model=NAME
    - name of model file and lists
- -c FILE, --config-file=FILE
    - read feature configuration from FILE

## --toCRFsuite
This option generates suitable input for CRFsuite from training and tagging data. Model name is required as the features and labels are translated to numbers and back. CRFsuite use its own bigram model.

# Usage examples

## train-tag

    # train
    cat input.txt | python3 huntag.py train --model=modelName --config-file=configs/hunchunk.krPatt.cfg
    # bigram-train
    cat input.txt | python3 huntag.py bigram-train --model=modelName
    # tag
    cat input.txt | python3 huntag.py tag --model=modelName --config-file=configs/hunchunk.krPatt.cfg

## CRFsuite usage

    # train toCRFsuite
    cat input.txt | python3 huntag.py train --toCRFsuite --model=modelName --config-file=configs/hunchunk.krPatt.cfg > modelName.CRFsuite.train
    # tag toCRFsuite
    cat input.txt | python3 huntag.py tag --toCRFsuite --model=modelName --config-file=configs/hunchunk.krPatt.cfg > modelName.CRFsuite.tag

## Debug features

    # train
    cat input.txt | python3 huntag.py train --model=modelName --config-file=configs/hunchunk.krPatt.cfg
    # most-informative-features
    cat input.txt | python3 huntag.py most-informative-features --model=modelName --config-file=configs/hunchunk.krPatt.cfg > modelName.mostInformativeFeatures
    # tag FeatureWeights
    cat input.txt | python3 huntag.py tag --printWeights 100 --model=modelName --config-file=configs/hunchunk.krPatt.cfg > modelNam.modelWeights



# Authors

HunTag3 is a massive overhaul and functional extension of the original HunTag codebase. HunTag3 was created by Balázs Indig.

HunTag was created by Gábor Recski and Dániel Varga. It is a reimplementation and generalization of a Named Entity Recognizer built by Dániel Varga and Eszter Simon.

# License

HunTag is made available under the GNU Lesser General Public License v3.0. If you received HunTag in a package that also contain the Hungarian training corpora for Named Entity Recoginition and chunking task, then please note that these corpora are derivative works based on the Szeged Treebank, and they are made available under the same restrictions that apply to the original Szeged Treebank

# Reference

If you use the tool, please cite the following paper:

Gábor Recski, Dániel Varga (2009): A Hungarian NP-chunker In: *The Odd Yearbook. ELTE SEAS Undergraduate Papers in Linguistics*. Budapest: ELTE School of English and American Studies. pp. 87-93

```
@article{Recski:2009a,
   author={Recski, G\'abor and D\'aniel Varga},
   title={{A Hungarian NP Chunker}},
   journal = {The Odd Yearbook. ELTE SEAS Undergraduate Papers in Linguistics},
   publisher = {ELTE {S}chool of {E}nglish and {A}merican {S}tudies},
   city = {Budapest},
   pages= {87--93},
   year={2009}
}
```

If you use some specialized version for Hungarian, please also cite the following paper:

Dóra Csendes, János Csirik, Tibor Gyimóthy and András Kocsor (2005): The Szeged Treebank. In: *Text, Speech and Dialogue. Lecture Notes in Computer Science* Volume 3658/2005, Springer: Berlin. pp. 123-131.

```
@inproceedings{Csendes:2005,
   author={Csendes, D{\'o}ra and Csirik, J{\'a}nos and Gyim{\'o}thy, Tibor and Kocsor, Andr{\'a}s},
   title={The {S}zeged {T}reebank},
   booktitle={Lecture Notes in Computer Science: Text, Speech and Dialogue},
   year={2005},
   pages={123-131},
   publisher={Springer}
}
```
