#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-
# Miscellaneous tools for HunTag

from operator import itemgetter
from collections import Counter, defaultdict
from itertools import count
import sys
import gzip


def sentence_iterator(input_stream):
    curr_sen = []
    curr_comment = None
    for line in input_stream:
        line = line.strip()
        # Comment handling
        if line.startswith('"""'):
            if len(curr_sen) == 0:  # Comment before sentence
                curr_comment = line
            else:  # Error: Comment in the middle of sentence
                print('ERROR: comments are only allowed before a sentence!', file=sys.stderr, flush=True)
                sys.exit(1)
        # Blank line handling
        elif len(line) == 0:
            if curr_sen:  # End of sentence
                yield curr_sen, curr_comment
                curr_sen = []
                curr_comment = None
            else:  # WARNING: Multiple blank line
                print('WARNING: wrong formatted sentences, only one blank line allowed!', file=sys.stderr, flush=True)
        else:
            curr_sen.append(line.split())
    if curr_sen:
        print('WARNING: No blank line before EOF!', file=sys.stderr, flush=True)
        yield curr_sen, curr_comment


def feature_names_to_indices(features, name_dict):
    for name, feature in features.items():
        feature.field_indices = [name_dict[f] for f in feature.fields]
    return features


def featurize_sentence(sen, features, feat_filter=lambda token_feats: token_feats, label_field=None):
    if label_field is None:  # Tagging
        sentence_feats = [[] for _ in sen]
    else:  # Training
        sentence_feats = [[fields[label_field]] for fields in sen]

    for feature in features.values():
        for c, feats in enumerate(feature.eval_sentence(sen)):
            sentence_feats[c] += feat_filter(feats)
    return sentence_feats


def use_featurized_sentence(sen, _, feat_filter=lambda token_feats: token_feats, label_field=None):
    if label_field is None:  # Tagging
        sentence_feats = [[] for _ in sen]
    else:  # Training
        sentence_feats = [[fields[label_field]] for fields in sen]

    for c, feats in enumerate(enumerate(sen)):
        sentence_feats[c] += feat_filter([feat for i, feat in enumerate(feats) if i != label_field])
    return sentence_feats


# Keeps Feature/Label-Number translation maps, for faster computations
class BookKeeper:
    def __init__(self, loadfromfile=None):
        self._counter = Counter()
        # Original source: (1.31) http://sahandsaba.com/thirty-python-language-features-and-tricks-you-may-not-know.html
        self._name_to_no = defaultdict(count().__next__)
        self.no_to_name = {}  # This is built only upon reading back from file
        if loadfromfile is not None:
            self._name_to_no.default_factory = count(start=self.load(loadfromfile)).__next__

    def num_of_names(self):
        return len(self._name_to_no)

    def makeno_to_name(self):
        self.no_to_name = {v: k for k, v in self._name_to_no.items()}
        assert len(self.no_to_name) == len(self._name_to_no)

    def cutoff(self, cutoff):
        to_delete = {self._name_to_no.pop(name) for name, counts in self._counter.items() if counts < cutoff}
        del self._counter
        new_name_no = {name: i for i, (name, _) in enumerate(sorted(self._name_to_no.items(), key=itemgetter(1)))}
        del self._name_to_no
        self._name_to_no = new_name_no
        return to_delete

    def get_no_tag(self, name):
        return self._name_to_no.get(name)  # Defaults to None

    def get_no_train(self, name):
        self._counter[name] += 1
        return self._name_to_no[name]  # Starts from 0 newcomers will get autoincremented value and stored

    def save(self, filename):
        with gzip.open(filename, mode='wt', encoding='UTF-8') as f:
            f.writelines('{}\t{}\n'.format(name, no) for name, no in sorted(self._name_to_no.items(),
                                                                            key=itemgetter(1)))

    def load(self, filename):
        no = 0  # Last no
        with gzip.open(filename, mode='rt', encoding='UTF-8') as f:
            for line in f:
                line = line.strip().split()
                name, no = line[0], int(line[1])
                self._name_to_no[name] = no
                self.no_to_name[no] = name
        return no
