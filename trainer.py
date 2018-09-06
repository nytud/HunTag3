#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

"""
trainer.py is a module of HunTag and is used to train maxent models
"""

import sys
from collections import Counter, defaultdict
from sklearn.externals import joblib
from scipy.sparse import csr_matrix
import numpy as np
from array import array
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDClassifier
# from sklearn.svm import SVC
# from sklearn.multiclass import OneVsRestClassifier

from tools import BookKeeper, sentence_iterator, featurize_sentence


class Trainer:
    def __init__(self, features, options):

        # Set clasifier algorithm here
        parameters = dict()  # dict(solver='lbfgs')
        solver = LogisticRegression

        # Possible alternative solvers:
        # parameters = {'loss':'modified_huber',  'n_jobs': -1}
        # solver = SGDClassifier

        # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        # parameters = {'kernel': 'rbf', 'probability': True}
        # solver = SVC

        # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        # parameters = {'kernel': 'linear', 'probability': True}
        # solver = OneVsRestClassifier(SVC(**parameters))  # XXX won't work because ** in parameters...

        self._model = solver(**parameters)
        self._dataSizes = options['data_sizes']
        self._tagField = options['tagField']
        self._modelFileName = options['modelFileName']
        self._parameters = options['trainParams']
        self._cutoff = options['cutoff']
        self._featCounterFileName = options['featCounterFileName']
        self._labelCounterFileName = options['labelCounterFileName']
        self._features = features

        self._tokCount = -1  # Index starts from 0

        self._rows = array(self._dataSizes['rows'])
        self._cols = array(self._dataSizes['cols'])
        self._data = array(self._dataSizes['data'])
        self._labels = array(self._dataSizes['labels'])
        self._sent_end = array(self._dataSizes['sentEnd'])  # Keep track of sentence boundaries
        self._matrix = None

        self._featCounter = BookKeeper()
        self._labelCounter = BookKeeper()
        self._usedFeats = None
        if 'usedFeats' in options and options['usedFeats']:
            self._usedFeats = {line.strip() for line in open(options['usedFeats'], encoding='UTF-8')}

    def save(self):
        print('saving model...', end='', file=sys.stderr, flush=True)
        joblib.dump(self._model, '{0}'.format(self._modelFileName), compress=3)
        print('done\nsaving feature and label lists...', end='', file=sys.stderr, flush=True)
        self._featCounter.save(self._featCounterFileName)
        self._labelCounter.save(self._labelCounterFileName)
        print('done', file=sys.stderr, flush=True)

    def _update_sent_end(self, sent_ends, row_nums):
        new_ends = array(self._dataSizes['sentEnd'])
        vbeg = 0
        for end in sent_ends:
            vend = -1
            for i, e in enumerate(row_nums[vbeg:]):
                if e <= end:
                    vend = vbeg + i
                else:
                    break
            if vend > 0:
                new_ends.append(vend)
                vbeg = vend + 1
        return new_ends

    def _convert_to_np_array(self):
        rows_np = np.array(self._rows, dtype=self._dataSizes['rows_np'])
        cols_np = np.array(self._cols, dtype=self._dataSizes['cols'])
        data_np = np.array(self._data, dtype=self._dataSizes['data'])
        labels_np = np.array(self._labels, dtype=self._dataSizes['labels'])
        del self._rows
        del self._cols
        del self._data
        del self._labels
        self._rows = rows_np
        self._cols = cols_np
        self._data = data_np
        self._labels = labels_np

    def _make_sparse_array(self, row_num, col_num):
        print('creating training problem...', end='', file=sys.stderr, flush=True)
        matrix = csr_matrix((self._data, (self._rows, self._cols)), shape=(row_num, col_num),
                            dtype=self._dataSizes['data'])
        del self._rows
        del self._cols
        del self._data
        print('done!', file=sys.stderr, flush=True)
        return matrix

    def cutoff_feats(self):
        self._convert_to_np_array()
        col_num = self._featCounter.num_of_names()
        if self._cutoff < 2:
            self._matrix = self._make_sparse_array(self._tokCount, col_num)
        else:
            print('discarding features with less than {0} occurences...'.format(self._cutoff), end='', file=sys.stderr,
                  flush=True)

            to_delete = self._featCounter.cutoff(self._cutoff)
            print('done!\nreducing training events by {0}...'.format(len(to_delete)), end='', file=sys.stderr,
                  flush=True)
            # ...that are not in featCounter anymore
            indices_to_keep_np = np.fromiter((ind for ind, featNo in enumerate(self._cols) if featNo not in to_delete),
                                             dtype=self._dataSizes['cols'])
            del to_delete

            # Reduce cols
            cols_np_new = self._cols[indices_to_keep_np]
            del self._cols
            self._cols = cols_np_new

            # Reduce data
            data_np_new = self._data[indices_to_keep_np]
            del self._data
            self._data = data_np_new

            # Reduce rows
            rows_np_new = self._rows[indices_to_keep_np]
            row_num_keep = np.unique(rows_np_new)
            row_num = row_num_keep.shape[0]
            col_num = indices_to_keep_np.max() + 1
            del self._rows
            self._rows = rows_np_new
            del indices_to_keep_np

            # Reduce labels
            labels_np_new = self._labels[row_num_keep]
            del self._labels
            self._labels = labels_np_new

            # Update sentence end markers
            new_end = self._update_sent_end(self._sent_end, row_num_keep)
            del self._sent_end
            self._sent_end = new_end
            del row_num_keep

            print('done!', file=sys.stderr, flush=True)
            matrix = self._make_sparse_array(row_num, col_num)
            print('updating indices...', end='', file=sys.stderr, flush=True)

            # Update rowNos
            rows, _ = matrix.nonzero()
            matrix_new = matrix[np.unique(rows), :]
            del matrix
            del rows

            # Update featNos
            _, cols = matrix_new.nonzero()
            self._matrix = matrix_new[:, np.unique(cols)]
            del matrix_new
            del cols

            print('done!', file=sys.stderr, flush=True)

    # Input need featurizing
    def get_events(self, data):
        print('featurizing sentences...', end='', file=sys.stderr, flush=True)
        sen_count = 0
        tok_index = -1  # Index starts from 0
        for sen, _ in sentence_iterator(data):
            sen_count += 1
            sentence_feats = featurize_sentence(sen, self._features)
            for c, tok in enumerate(sen):
                tok_index += 1
                tok_feats = sentence_feats[c]
                if self._usedFeats:
                    tok_feats = [feat for feat in tok_feats if feat in self._usedFeats]
                self._add_context(tok_feats, tok[self._tagField], tok_index)
            self._sent_end.append(tok_index)
            if sen_count % 1000 == 0:
                print('{0}...'.format(str(sen_count)), end='', file=sys.stderr, flush=True)

        self._tokCount = tok_index + 1
        print('{0}...done!'.format(str(sen_count)), file=sys.stderr, flush=True)

    # Already featurized input
    def get_events_from_file(self, data):
        tok_index = -1  # Index starts from 0
        for line in data:
            line = line.strip()
            if len(line) > 0:
                tok_index += 1
                line = line.split()
                label, feats = line[0], line[1:]
                self._add_context(feats, label, tok_index)
            self._sent_end.append(tok_index)
        self._tokCount = tok_index + 1

    def _add_context(self, tok_feats, label, cur_tok):
        rows_append = self._rows.append
        cols_append = self._cols.append
        data_append = self._data.append

        # Features are sorted to ensure identical output no matter where the features are coming from
        for featNumber in {self._featCounter.get_no_train(feat) for feat in sorted(tok_feats)}:
            rows_append(cur_tok)
            cols_append(featNumber)
            data_append(1)

        self._labels.append(self._labelCounter.get_no_train(label))

    # Counting zero elements can be really slow...
    def most_informative_features(self, output_stream=sys.stdout, n=-1, count_zero=False):
        # Compute min(P(feature=value|label1), for any label1)/max(P(feature=value|label2), for any label2)
        # (using contitional probs using joint probabilities) as in NLTK (Bird et al. 2009):
        # P(feature=value|label) = P(feature=value, label)/P(label)
        # P(feature=value, label) = C(feature=value, label)/C(feature=value)
        # P(label) = C(label)/sum_i(C(label_i))
        #
        # P(feature=value|label) = (C(feature=value, label)/C(feature=value))/(C(label)/sum_i(C(label_i))) =
        # (C(feature=value, label)*sum_i(C(label_i)))/(C(feature=value)*C(label))
        #
        # min(P(feature=value|label1), for any label1)/max(P(feature=value|label2), for any label2) =
        #
        # min((C(feature=value, label1)*sum_i(C(label_i)))/(C(feature=value)*C(label1)), for any label1)/
        # max((C(feature=value, label2)*sum_i(C(label_i)))/(C(feature=value)*C(label2)), for any label2) =
        #
        # (sum_i(C(label_i))/C(feature=value))*min(C(feature=value, label1)/C(label1)), for any label1)/
        # (sum_i(C(label_i))/C(feature=value))*max(C(feature=value, label2)/C(label2)), for any label2) =
        #
        # min(C(feature=value, label1)/C(label1), for any label1)/
        # max(C(feature=value, label2)/C(label2), for any label2)
        matrix = self._matrix  # For easiser handling
        self._featCounter.makeno_to_name()
        self._labelCounter.makeno_to_name()
        featno_to_name = self._featCounter.no_to_name
        labelno_to_name = self._labelCounter.no_to_name
        labels = self._labels  # indexed by token rows (row = token number, column = feature number)
        feat_val_counts = defaultdict(Counter)  # feat, val -> label: count

        if count_zero:
            # Every index (including zeros to consider negative correlation)
            for feat in range(matrix.shape[1]):
                for tok in range(matrix.shape[0]):
                    feat_val_counts[feat, matrix[tok, feat]][labels[tok]] += 1
        else:
            matrix = matrix.tocoo()
            # Every nonzero index
            for tok, feat, val in zip(matrix.row, matrix.col, matrix.data):
                feat_val_counts[feat, val][labels[tok]] += 1
        del matrix

        # (C(label2), for any label2)
        label_counts = Counter()
        for k, v in zip(*np.unique(self._labels, return_counts=True)):
            label_counts[k] = v

        num_of_labels = len(label_counts)
        maxprob = defaultdict(lambda: 0.0)
        minprob = defaultdict(lambda: 1.0)
        features = set()
        # For every (feature, val) touple (that has nonzero count)
        for feature, counts in feat_val_counts.items():
            # For every label label...
            features.add(feature)
            for label, count in counts.items():
                # prob can only be 0 if the nominator is 0, but this case is already filtered in the Counter...
                prob = count/label_counts[label]
                maxprob[feature] = max(prob, maxprob[feature])
                minprob[feature] = min(prob, minprob[feature])

        # Convert features to a list, & sort it by how informative features are.
        """
        From NTLK docs:
        For the purpose of this function, the
        informativeness of a feature ``(fname,fval)`` is equal to the
        highest value of P(fname=fval|label), for any label, divided by
        the lowest value of P(fname=fval|label), for any label:

        |  max[ P(fname=fval|label1) / P(fname=fval|label2) ]
        """
        print('"Feature name"=Value (True/False)', 'Sum of occurences', 'Counts per label', 'Probability per label',
              'Max prob.:Min prob.=Ratio:1.0', sep='\t', file=output_stream)  # Print header (legend)
        # To avoid division by zero...
        for feature in sorted(features, key=lambda feature_: minprob[feature_]/maxprob[feature_])[:n]:
            sum_occurences = sum(feat_val_counts[feature].values())
            if len(feat_val_counts[feature]) < num_of_labels:
                ratio = 'INF'
            else:
                ratio = maxprob[feature]/minprob[feature]
            # NLTK notation
            # print('{0:50} = {1:} {2:6} : {3:-6} = {4} : 1.0'.format(featno_to_name(feature[0]), feature[1],
            #                                                                maxprob[feature],
            #                                                                minprob[feature], ratio))
            # More detailed notation
            print('"{0:50s}"={1}\t{2}\t{3}\t{4}\t{5:6}:{6:-6}={7}:1.0'.format(
                featno_to_name[feature[0]],
                bool(feature[1]),
                sum_occurences,
                '/'.join(('{0}:{1}'.format(labelno_to_name[l], c)
                          for l, c in feat_val_counts[feature].items())),
                '/'.join(('{0}:{1:.8f}'.format(labelno_to_name[l], c/label_counts[l])
                          for l, c in feat_val_counts[feature].items())),
                maxprob[feature], minprob[feature], ratio), file=output_stream)

    def to_crfsuite(self, output_stream=sys.stdout):
        self._featCounter.makeno_to_name()
        self._labelCounter.makeno_to_name()
        featno_to_name = self._featCounter.no_to_name
        labelno_to_name = self._labelCounter.no_to_name
        sent_end = self._sent_end
        matrix = self._matrix.tocsr()
        labels = self._labels
        beg = 0
        for end in sent_end:
            for row in range(beg, end + 1):
                print('{0}\t{1}'.format(labelno_to_name[labels[row]],
                                        '\t'.join(featno_to_name[col].replace(':', 'colon')
                                                  for col in matrix[row, :].nonzero()[1])),
                      file=output_stream)
            print(file=output_stream)  # Sentence separator blank line
            beg = end + 1

    def train(self):
        print('training with option(s) "{0}"...'.format(self._parameters), end='', file=sys.stderr, flush=True)
        _ = self._model.fit(self._matrix, self._labels)
        print('done', file=sys.stderr, flush=True)
