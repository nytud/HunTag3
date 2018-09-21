HunTag3 - A sequential tagger for NLP combining the Scikit-learn/LinearRegressionClassifier linear classifier and Hidden Markov Models.  
  
Based on training data, HunTag3 can perform any kind of sequential sentence  
tagging and has been used for NP chunking and Named Entity Recognition for English and Hungarian.  
  
HungTag3 is the official successor of [HunTag](https://github.com/recski/HunTag) project.  
(See git tags for past major milestones.)  
  
# Highlights  
  
- **NEW:** Has a REST API for tagging files with a pretrained model  
- Using [NumPy](http://www.numpy.org/)/[SciPy.sparse](http://scipy.org/) arrays  
- Runabble [YAML](http://www.yaml.org/) configuration  
- Independent unigram model from [SciKit-learn](http://scikit-learn.org/stable/) (LinearRegressionClassifier)  
- Selectable bi- or trigram transition model (Can be trained separately)  
- Features written in native Python code and lexicons are enabled  
- Can work with featurized input and can generate featurized output for other taggers  
- Consumes minimal memory possible  
- [NLTK.classify.naivebayes-like](http://www.nltk.org/api/nltk.classify.html#nltk.classify.naivebayes.NaiveBayesClassifier.most_informative_features) *Most Informative Features* function for examining features quality  
- Able to write the unigram featuret weights  
  
# Requirements  
  
- See requirements.txt (pip install -r requirements.txt)  
- Optional: CRFsuite  
- Minimum 8 GB RAM recomended for training  
  
# Data format  
  
- Input data must be a tab-separated file (TSV) with one word per line and **the first line must contain the field names**  
- An empty line to mark sentence boundaries  
- Each line must contain the same number of fields (except the first line, which must name a field for the output)  
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
  
 default: cutoff: 1  #  1 if not set radius: 5  # -1 if not set  
There are three type of features:  
- **Sentence**: The input is aggregated sentence-wise into a list, and this list is then passed to the feature function. This function should return a list consisting of one feature string for each of the tokens of the sentence  
- **Token**: Get a token, and returns a feature list or bool for each given token independently  
- **Lexicon**: The specified token field is matched against this lexicon file and ubstitutes each token with its features from the lexicon  
  
For each feature mandatory fields are the following:  
- **name**: name of the feature as appears in the output (featurized input, most-informative-features, etc.)  
- **type**: sentence/token/lexicon  
- **action_name**: Refers to the features.py function name or the lexicon file  
- **fields**: Refers to the field names of the input (must present in the first line of the input), that the feature use (only *sentence* type features allowed to have more values here separated by comma. Lexicon features must supply the field of the token here)  
- **radius**: **(Only for sentence type features)** add the features of each corresponding token to the feature list of all the token its given length radius (independent from the feature)  
- **options**: **(Only for sentence type features)** Here one can enumerate all options that the corresponding feature need (see feature documentation in features.py)   
  
See configs folder for examples on the format.  
  
# Usage  
HunTag may be run in any of the following modes (see startHuntag.sh for overview and *huntag_main.py --help* for details):  
  
## train and train-featurize  
Used to train a model or just featurize given a training corpus with a set of feature functions. When run in TRAIN mode, HunTag creates three files, one containing the model and two listing features and labels and the integers they are mapped to when passed to the learner. With the --model option set to NAME, the three files will be stored under NAME.model, NAME.featureNumbers.gz and NAME.labelNumbers.gz respectively.  
  
 cat TRAINING_DATA | python3 huntag_main.py train OPTIONSor  
   huntag_main.py train -i TRAINING_DATA OPTIONS  
  
Mandatory options:  
- -c FILE, --config-file=FILE  
  - read feature configuration from FILE  
- -m NAME, --model=NAME  
  - name of model and lists  
  
Non-mandatory options:  
- -i INPUT, --input=INPUT  
   - input is taken from INPUT file instead of STDIN  
  
## transmodel-train  
Used to train a transition model (from a bigram or trigram language model) using a given field of the training data  
  
 cat TRAINING_DATA | python3 huntag_main.py transmodel-train OPTIONSor  
   huntag_main.py transmodel-train -i TRAINING_DATA OPTIONS  
  
Mandatory options:  
- -m NAME, --model=NAME  
  - name of model file and lists  
- --trans-model-order [2 or 3, default: 3]  
 - order of the transition model (bigram or trigram)  
  
Non-mandatory options:  
- -i INPUT, --input=INPUT  
   - input is taken from INPUT file instead of STDIN  
  
## tag or tag-featurize  
Used to tag or just featurize the input. Given a maxent model providing the value P(l|w) for all labels l and words (set of feature values) w, and a transition model supplying P(l|l0) for all pairs of labels, HunTag will assign to each sentence the most likely label sequence.  
  
 cat INPUT | python3 huntag_main.py tag OPTIONSor  
   huntag_main.py tag -i INPUT OPTIONS  
  
Mandatory options:  
- -m NAME, --model=NAME  
  - name of model file and lists  
- -c FILE, --config-file=FILE  
  - read feature configuration from FILE  
  
Non-mandatory options:  
- -l L, --language-model-weight=L  
  - set weight of the language model to L (default is 1)  
- -i INPUT, --input=INPUT  
   - input is taken from INPUT file instead of STDIN  
- -o OUTPUT, --output=OUTPUT  
   - output is written to OUTPUT file instead of STDOUT  
  
## most-informative-features  
Generates a feature ranking by counting label probabilities (for each label) and frequency per feature (correlations with labels) and sort them in decreasing order of confidence and frequency. This output is usefull for inspecting features quality.  
  
 cat TRAINING_DATA | python3 huntag_main.py most-informative-features OPTIONS > modelName.most_informative_featuresor  
   huntag_main.py most-informative-features -i TRAINING_DATA  OPTIONS  
  
Mandatory options:  
- -m NAME, --model=NAME  
  - name of model file and lists  
- -c FILE, --config-file=FILE  
  - read feature configuration from FILE  
  
Non-mandatory options:  
- -i INPUT, --input=INPUT  
   - input is taken from INPUT file instead of STDIN  
- -o OUTPUT, --output=OUTPUT  
   - output is written to OUTPUT file instead of STDOUT  
  
## tag --print-weights N  
Usefull for inspecting feature weights (per label) assigned by the MaxEnt learner. (As name suggests, training must happen before tagging.)  
Negative weights mean negative correlation, which is also usefull.  
  
 python3 huntag.py tag --print-weights N OPTIONS > modelName.modelWeightsor  
   huntag.py tag --print-weights N OPTIONS -o modelName.modelWeights  
  
Mandatory options:  
- N is the number of features to print (default: 100)  
- -m NAME, --model=NAME  
  - name of model file and lists  
- -c FILE, --config-file=FILE  
  - read feature configuration from FILE  
  
Non-mandatory options:  
- -o OUTPUT, --output=OUTPUT  
   - output is written to OUTPUT file instead of STDOUT  
  
## train-featurize and tag-feturize  
This options generate suitable input for CRFsuite from training and tagging data. Model name is required as the features and labels are translated to numbers and back. CRFsuite use its own bigram model.  
  
## REST API    
 - Edit the `model_name` and `cfg_file` in `huntag_rest.py` and run the file `python3 huntag_rest.py` - Note: Setup an appropriate Python3 virtualenv in conjunction with a WSGI server for performance.    
- The clients should ask the following or similar:    


    r = requests.post('http://127.0.0.1:5000/tag_file', files={'file':open('test.maxnp.hfst', encoding='UTF-8')})

# Usage examples  
  
A 100 token long example can be found in the git repository for clarifying the format to be used.  
  
## Basic usage: train-tag    
    
    # train
    cat input.txt | python3 huntag_main.py train --model=modelName --config-file=configs/maxnp.szeged.hfst.yaml
    # transmodel-train
    cat input.txt | python3 huntag_main.py transmodel-train --model=modelName  # --trans-model-order [2 or 3, default: 3]
    # tag
    cat input.txt | python3 huntag_main.py tag --model=modelName --config-file=configs/maxnp.szeged.hfst.yaml ## Featurizing input (eg. for CRFsuite)
    
## Advanced usage (for example with CRFsuite):

    # train-featurize
    cat input.txt | python3 huntag_main.py train-featurize --model=modelName --config-file=configs/maxnp.szeged.hfst.yaml > modelName.CRFsuite.train
    # tag-featurize
    cat input.txt | python3 huntag_main.py tag-featurize --model=modelName --config-file=configs/maxnp.szeged.hfst.yaml > modelName.CRFsuite.tag

## Debuging features:

    # most-informative-features
    cat input.txt | python3 huntag_main.py most-informative-features --model=modelName --config-file=configs/maxnp.szeged.hfst.yaml > modelName.most_informative_features 
    # tag FeatureWeights
    cat input.txt | python3 huntag_main.py tag --print-weights 100 --model=modelName --config-file=configs/maxnp.szeged.hfst.yaml > modelNam.modelWeights

# Authors  
  
HunTag3 is a massive overhaul, cleanup and functional extension of the original HunTag idea and codebase. HunTag3 was created by Balázs Indig with contributions from Márton Miháltz.  
  
HunTag was created by Gábor Recski and Dániel Varga. It is a reimplementation and generalization of a Named Entity Recognizer built by Dániel Varga and Eszter Simon.  
  
The patch for Liblinear (to lower memory usage) was created by Attila Zséder. See link for deatils: http://www.csie.ntu.edu.tw/~cjlin/liblinear/faqfiles/python_datastructures.html  
  
# License  
  
HunTag3 is made available under the GNU Lesser General Public License v3.0. If you received HunTag3 in a package that also contain the Hungarian training corpora for named-entity recognition or chunking task, then please note that these corpora are derivative works based on the Szeged Treebank, and they are made available under the same restrictions that apply to the original Szeged Treebank  
  
# Reference  
  
This tool is also [integrated](https://github.com/dlt-rilmta/hunlp-GATE)  
into the [e-magyar](http://www.e-magyar.hu) language processing system. It is called emNER.  
  
If you use the tool, please cite the following paper:  
  
[István Endrédy and Balázs Indig (2015): *HunTag3: a general-purpose, modular sequential tagger -- chunking phrases in English and maximal NPs and NER for Hungarian*  
In: Zygmunt Vetulani; Joseph Mariani (eds.) 7th Language & Technology Conference: Human Language Technologies as a Challenge for Computer Science and Linguistics. (2015.11.27-2015.11.30, Poznań, Poland)  
"Poznań: Uniwersytet im. Adama Mickiewicza w Poznaniu" 558 p. ISBN:978-83-932640-8-7 pp. 213-218.](http://ltc.amu.edu.pl/book/papers/TAN2-2.pdf)  
  
```  
@inproceedings{HunTag3,  
 title       = {{HunTag3:} a general-purpose, modular sequential tagger -- chunking phrases in {English and maximal NPs and NER for Hungarian}}, author      = {Endr\'edy, Istv\'an and Indig, Bal\'azs}, booktitle   = {7th {L}anguage \& {T}echnology {C}onference, {Human Language Technologies as a Challenge for Computer Science and Linguistics (LTC '15)}}, year        = {2015}, month       = {November},  publisher   = {Pozna\'n: {U}niwersytet im. {Adama Mickiewicza w Poznaniu}},  
 isbn        = {978-83-932640-8-7}, pages       = {213-218}, address     = {{P}ozna\'n, {P}oland}}  
```  
  
If you use some specialized version for Hungarian, please also cite the following paper:  
  
Dóra Csendes, János Csirik, Tibor Gyimóthy and András Kocsor (2005): The Szeged Treebank. In: *Text, Speech and Dialogue. Lecture Notes in Computer Science* Volume 3658/2005, Springer: Berlin. pp. 123-131.  
  
```  
@inproceedings{Csendes:2005,  
 author={Csendes, D{\'o}ra and Csirik, J{\'a}nos and Gyim{\'o}thy, Tibor and Kocsor, Andr{\'a}s}, title={The {S}zeged {T}reebank}, booktitle={Lecture Notes in Computer Science: Text, Speech and Dialogue}, year={2005}, pages={123-131}, publisher={Springer}}  
```