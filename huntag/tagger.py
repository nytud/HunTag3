#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import sys
import joblib
from scipy.sparse import csr_matrix

from .tools import BookKeeper, featurize_sentence, use_featurized_sentence, bind_features_to_indices, \
    load_options_and_features
from .transmodel import TransModel
from .argparser import valid_file


class Tagger:
    pass_header = True

    def __init__(self, opts, source_fields=None, target_fields=None):
        if 'cfg_file' in opts:
            opts['cfg_file'] = valid_file(opts['cfg_file'])  # Validate config file!
        self.features, self.source_fields, self.target_fields, options = \
            load_options_and_features(opts, source_fields, target_fields)

        self._tag_field = None

        self._data_sizes = options['data_sizes']

        if options['task'] not in {'print-weights', 'tag-featurize'}:
            print('loading transition model...', end='', file=sys.stderr, flush=True)
            self._trans_probs = TransModel.load_from_file(valid_file(options['transmodel_filename']))
            print('done', file=sys.stderr, flush=True)
        else:
            self._trans_probs = None

        print('loading observation model...', end='', file=sys.stderr, flush=True)
        self._model = joblib.load(valid_file(options['model_filename']))
        self._feat_counter = BookKeeper(valid_file(options['featcounter_filename']))
        self._label_counter = BookKeeper(valid_file(options['labelcounter_filename']))
        print('done', file=sys.stderr, flush=True)

        # Set functions according to task...
        if options.get('inp_featurized', False):
            self._featurize_sentence_fun = use_featurized_sentence
            self._format_output = self._add_tagging_featurized
            self._tag_fun = self.tag_by_feat_number
        else:
            self._featurize_sentence_fun = featurize_sentence
            if options.get('task') == 'tag-featurize':  # print features
                self._format_output = self._feat_counter.no_to_name
                self._tag_fun = self._print_features
            else:  # tag sentences
                self._format_output = self._add_tagging_normal
                self._tag_fun = self.tag_by_feat_number

    def _get_tag_probs_by_pos(self, feat_numbers):
        rows, cols, data = [], [], []
        for rownum, featNumberSet in enumerate(feat_numbers):
            for featNum in featNumberSet:
                rows.append(rownum)
                cols.append(featNum)
                data.append(1)
        contexts = csr_matrix((data, (rows, cols)), shape=(len(feat_numbers), self._feat_counter.num_of_names()),
                              dtype=self._data_sizes['data_np'])
        tagprobs_by_pos = [{self._label_counter.no_to_name[i]: prob for i, prob in enumerate(prob_dist)}
                           for prob_dist in self._model.predict_proba(contexts)]
        return tagprobs_by_pos

    @staticmethod
    def _add_tagging_featurized(_, best_tagging, __):
        return [[label] for label in best_tagging]

    @staticmethod
    def _add_tagging_normal(sent, best_tagging, tag_index):
        for tok, label in zip(sent, best_tagging):
            tok.insert(tag_index, label)
        return sent

    def tag_by_feat_number(self, sen, feat_numbers, add_tagging, tag_index):
        best_tagging = self._trans_probs.tag_sent(self._get_tag_probs_by_pos(feat_numbers))
        return add_tagging(sen, best_tagging, tag_index)  # Add tagging to sentence

    @staticmethod
    def _print_features(_, feat_numbers, featno_to_name, __):
        return [[featno_to_name[featNum].replace(':', 'colon') for featNum in featNumberSet]
                for featNumberSet in feat_numbers]

    def prepare_fields(self, field_names):
        target_fields_len = len(self.target_fields)
        if target_fields_len != 1:
            print('ERROR: Wrong number of target fields are specified ({0})! '
                  'TAGGING REQUIRES ONLY ONE TAG FIELD!'.
                  format(target_fields_len), file=sys.stderr, flush=True)
            sys.exit(1)
        self._tag_field = field_names[self.target_fields[0]]
        return bind_features_to_indices(self.features, {k: v for k, v in field_names.items()
                                                        if k != self._tag_field and v != self._tag_field})

    def process_sentence(self, sen, features_bound_to_column_ids):
        sen_feats = self._featurize_sentence_fun(sen, features_bound_to_column_ids)
        get_no_tag = self._feat_counter.get_no_tag
        # Get Sentence Features translated to numbers and contexts in two steps
        feat_numbers = [{get_no_tag(feat) for feat in feats if get_no_tag(feat) is not None} for feats in sen_feats]
        return self._tag_fun(sen, feat_numbers, self._format_output, self._tag_field)

    def print_weights(self, output_stream, n=100):
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
