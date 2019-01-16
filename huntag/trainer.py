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

from huntag.tools import BookKeeper, featurize_sentence, use_featurized_sentence, bind_features_to_indices


class Trainer:
    def __init__(self, features, options, source_fields=None, target_fields=None):

        # Set clasifier algorithm here
        parameters = {'solver': 'lbfgs', 'multi_class': 'auto'}
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

        self._cutoff = options['cutoff']
        self._parameters = options['train_params']
        self._model = solver(**parameters)

        self.features = features

        # Field names for e-magyar TSV
        if source_fields is None:
            source_fields = {}

        if target_fields is None:
            target_fields = []

        self.source_fields = source_fields
        self.target_fields = target_fields

        self.source_fields.union({field for feat in features.values() for field in feat.fields})
        self.target_fields = []
        self._tag_field_name = options['gold_tag_field']

        self._model_file_name = options['model_filename']
        self._feat_counter_file_name = options['featcounter_filename']
        self._label_counter_file_name = options['labelcounter_filename']

        if options['inp_featurized']:
            self._featurize_sentence_fun = use_featurized_sentence
        else:
            self._featurize_sentence_fun = featurize_sentence

        self._tok_count = -1  # Index starts from 0

        self._data_sizes = options['data_sizes']
        self._rows = array(self._data_sizes['rows'])
        self._cols = array(self._data_sizes['cols'])
        self._data = array(self._data_sizes['data'])
        self._labels = array(self._data_sizes['labels'])
        self._sent_end = array(self._data_sizes['sent_end'])  # Keep track of sentence boundaries
        self._matrix = None

        self._feat_counter = BookKeeper()
        self._label_counter = BookKeeper()

        self._feat_filter = lambda token_feats: token_feats
        feat_filename = options.get('used_feats')
        if feat_filename is not None:
            used_feats = {line.strip() for line in open(feat_filename, encoding='UTF-8')}
            self._feat_filter = lambda token_feats: [feat for feat in token_feats if feat in used_feats]
            self._tag_field = 0  # Always the first field!

    def save(self):
        print('saving model...', end='', file=sys.stderr, flush=True)
        joblib.dump(self._model, '{0}'.format(self._model_file_name), compress=3)
        print('done\nsaving feature and label lists...', end='', file=sys.stderr, flush=True)
        self._feat_counter.save(self._feat_counter_file_name)
        self._label_counter.save(self._label_counter_file_name)
        print('done', file=sys.stderr, flush=True)

    def _update_sent_end(self, sent_ends, row_nums):
        new_ends = array(self._data_sizes['sent_end'])
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
        rows_np = np.array(self._rows, dtype=self._data_sizes['rows_np'])
        cols_np = np.array(self._cols, dtype=self._data_sizes['cols'])
        data_np = np.array(self._data, dtype=self._data_sizes['data'])
        labels_np = np.array(self._labels, dtype=self._data_sizes['labels'])
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
                            dtype=self._data_sizes['data'])
        del self._rows
        del self._cols
        del self._data
        print('done!', file=sys.stderr, flush=True)
        return matrix

    def cutoff_feats(self):
        self._tok_count += 1  # This actually was the token index which starts from 0...
        self._convert_to_np_array()
        col_num = self._feat_counter.num_of_names()
        if self._cutoff < 2:  # Keep all...
            self._matrix = self._make_sparse_array(self._tok_count, col_num)
        else:
            print('discarding features with less than {0} occurences...'.format(self._cutoff), end='', file=sys.stderr,
                  flush=True)

            to_delete = self._feat_counter.cutoff(self._cutoff)
            print('done!\nreducing training events by {0}...'.format(len(to_delete)), end='', file=sys.stderr,
                  flush=True)
            # ...that are not in featCounter anymore
            indices_to_keep_np = np.fromiter((ind for ind, featNo in enumerate(self._cols) if featNo not in to_delete),
                                             dtype=self._data_sizes['cols'])
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

    def prepare_fields(self, field_names):
        self._tag_field = field_names.get(self._tag_field_name)  # Bind tag field separately as it has no feature
        return bind_features_to_indices(self.features, {k: v for k, v in field_names.items()
                                                        if k != self._tag_field_name and v != self._tag_field_name})

    def process_sentence(self, sen, features):
        """
        Read input data in variable forms
        :param sen: one token per elem
        :param features: the features bound to columns
        :return: dummy list of tokens which are list of features
        """
        for label, *feats in self._featurize_sentence_fun(sen, features, self._feat_filter, self._tag_field):
            self._tok_count += 1
            self._add_context(feats, label, self._tok_count)
        self._sent_end.append(self._tok_count)
        return [[]]  # Dummy

    def _add_context(self, tok_feats, label, cur_tok):
        rows_append = self._rows.append
        cols_append = self._cols.append
        data_append = self._data.append

        # Features are sorted to ensure identical output no matter where the features are coming from
        for featNumber in {self._feat_counter.get_no_train(feat) for feat in sorted(tok_feats)}:
            rows_append(cur_tok)
            cols_append(featNumber)
            data_append(1)

        self._labels.append(self._label_counter.get_no_train(label))

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
        self._feat_counter.makeno_to_name()
        self._label_counter.makeno_to_name()
        featno_to_name = self._feat_counter.no_to_name
        labelno_to_name = self._label_counter.no_to_name
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

    def write_featurized_input(self, output_stream=sys.stdout):
        self._feat_counter.makeno_to_name()
        self._label_counter.makeno_to_name()
        featno_to_name = self._feat_counter.no_to_name
        labelno_to_name = self._label_counter.no_to_name
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
