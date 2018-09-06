#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import sys
import os
from sklearn.externals import joblib
from scipy.sparse import csr_matrix

from tools import sentence_iterator, featurize_sentence, BookKeeper


class Tagger:
    def __init__(self, features, trans_model, options):
        self._features = features
        self._data_sizes = options['data_sizes']
        self._trans_probs = trans_model
        print('loading observation model...', end='', file=sys.stderr, flush=True)
        self._model = joblib.load('{0}'.format(options['modelFileName']))
        self._feat_counter = BookKeeper(options['featCounterFileName'])
        self._label_counter = BookKeeper(options['labelCounterFileName'])
        print('done', file=sys.stderr, flush=True)

    def print_weights(self, n=100, output_stream=sys.stdout):
        coefs = self._model.coef_
        labelno_to_name = self._label_counter.no_to_name
        featno_to_name = self._feat_counter.no_to_name
        sorted_feats = sorted(featno_to_name.items())
        for i, label in sorted(labelno_to_name.items()):
            columns = ['{0}:{1}'.format(w, feat) for w, (no, feat) in sorted(zip(coefs[i, :], sorted_feats),
                                                                             reverse=True)]
            print('{0}\t{1}'.format(label, '\t'.join(columns[:n])), file=output_stream)  # Best
            # Worst -> Negative correlation
            print('{0}\t{1}'.format(label, '\t'.join(sorted(columns[-n:], reverse=True))), file=output_stream)

    def tag_features(self, data):
        sen_feats = []
        sen_count = 0
        for line in data:
            line = line.strip()
            if len(line) == 0:
                sen_count += 1
                tagging = self._tag_sen_feats(sen_feats)
                yield [[tag] for tag in tagging]
                sen_feats = []
                if sen_count % 1000 == 0:
                    print('{0}...'.format(sen_count), end='', file=sys.stderr, flush=True)
            sen_feats.append(line.split())
        print('{0}...done'.format(sen_count), file=sys.stderr, flush=True)

    def tag_dir(self, dir_name):
        for fn in os.listdir(dir_name):
            print('processing file {0}...'.format(fn), end='', file=sys.stderr, flush=True)
            for sen, _ in self.tag_corp(open(os.path.join(dir_name, fn), encoding='UTF-8')):
                yield sen, fn

    def tag_corp(self, input_stream=sys.stdin):
        sen_count = 0
        for sen, comment in sentence_iterator(input_stream):
            sen_count += 1
            sen_feats = featurize_sentence(sen, self._features)
            best_tagging = self._tag_sen_feats(sen_feats)
            tagged_sen = [tok + [best_tagging[c]] for c, tok in enumerate(sen)]  # Add tagging to sentence
            yield tagged_sen, comment
            if sen_count % 1000 == 0:
                print('{0}...'.format(sen_count), end='', file=sys.stderr, flush=True)
        print('{0}...done'.format(sen_count), file=sys.stderr, flush=True)

    def _get_tag_probs_by_pos(self, sen_feats):
        # Get Sentence Features translated to numbers and contexts in two steps
        get_no_tag = self._feat_counter.get_no_tag
        feat_numbers = [{get_no_tag(feat) for feat in feats if get_no_tag(feat) is not None} for feats in sen_feats]

        rows = []
        cols = []
        data = []
        for rownum, featNumberSet in enumerate(feat_numbers):
            for featNum in featNumberSet:
                rows.append(rownum)
                cols.append(featNum)
                data.append(1)
        contexts = csr_matrix((data, (rows, cols)), shape=(len(feat_numbers), self._feat_counter.num_of_names()),
                              dtype=self._data_sizes['dataNP'])
        tagprobs_by_pos = [{self._label_counter.no_to_name[i]: prob for i, prob in enumerate(prob_dist)}
                           for prob_dist in self._model.predict_proba(contexts)]
        return tagprobs_by_pos

    def to_crfsuite(self, input_stream, output_stream=sys.stdout):
        sen_count = 0
        get_no_tag = self._feat_counter.get_no_tag
        featno_to_name = self._feat_counter.no_to_name
        for sen, comment in sentence_iterator(input_stream):
            sen_count += 1
            sen_feats = featurize_sentence(sen, self._features)
            # Get Sentence Features translated to numbers and contexts in two steps
            for featNumberSet in ({get_no_tag(feat) for feat in feats if get_no_tag(feat) is not None}
                                  for feats in sen_feats):
                print('\t'.join(featno_to_name[featNum].replace(':', 'colon') for featNum in featNumberSet),
                      file=output_stream)
            print(file=output_stream)  # Sentence separator blank line
            if sen_count % 1000 == 0:
                print('{0}...'.format(str(sen_count)), end='', file=sys.stderr, flush=True)
        print('{0}...done'.format(str(sen_count)), file=sys.stderr, flush=True)

    def _tag_sen_feats(self, sen_feats):
        return self._trans_probs.tag_sent(self._get_tag_probs_by_pos(sen_feats))
