#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-
"""
feature.py is a module of HunTag. The Feature class is used for representing
a feature type and calculating its value for some input. Feature instances are
created by the getFeatureSet function in huntag_main.py.
"""

import sys

from huntag import features


class Feature:
    def __init__(self, kind, name, action_name, fields, radius, cutoff, options):
        self.kind = kind
        self.name = name
        self.action_name = action_name
        self.fields = fields
        self.field_indices = None
        self.radius = int(radius)
        self.cutoff = int(cutoff)
        self.options = options
        if self.kind in ('token', 'lex') and len(self.fields) != 1:
            print('Error: Feature (token, lex) "{0}" field count must be one not {1}!'.format(self.name, self.fields),
                  file=sys.stderr, flush=True)
            sys.exit(1)
        if self.kind == 'lex':
            if len(self.options) > 0:
                print('Lexicon features do not yet support options', file=sys.stderr, flush=True)
                sys.exit(1)
            self.lexicon = Lexicon(action_name)  # Load input file

        elif self.kind in ('token', 'sentence'):
            function_name = '{0}_{1}'.format(self.kind, self.action_name)
            if function_name not in features.__dict__:
                print('Unknown operator named {0}\n'.format(self.action_name), file=sys.stderr, flush=True)
                sys.exit(1)
            self.function = features.__dict__[function_name]
        else:
            print('Unknown kind named {0}'.format(self.kind), file=sys.stderr, flush=True)
            sys.exit(1)

    def eval_sentence(self, sentence):
        if self.kind == 'token':
            # Pick the relevant fields (label can be not just the last field)
            feat_vec = [self.function(word[self.field_indices[0]], self.options) for word in sentence]
        elif self.kind == 'lex':
            # Word will be substituted by its features from the Lexicon
            # self.fields denote the column of the word
            feat_vec = self.lexicon.lex_eval_sentence([word[self.field_indices[0]] for word in sentence])
        elif self.kind == 'sentence':
            feat_vec = self.function(sentence, self.field_indices, self.options)
        else:
            print('eval_sentence: Unknown kind named {0}'.format(self.kind), file=sys.stderr, flush=True)
            sys.exit(1)
        return self._multiply_features(sentence, feat_vec)

    def _multiply_features(self, sentence, feat_vec):
        sentence_len = len(sentence)
        multiplied_feat_vec = [[] for _ in range(sentence_len)]
        for c in range(sentence_len):
            # Iterate the radius, but keep the bounds of the list!
            for pos in range(max(c - self.radius, 0),
                             min(c + self.radius + 1, sentence_len)):
                # All the feature that assigned for a token
                for feat in feat_vec[pos]:
                    if feat != 0:  # XXX feat COULD BE string...
                        multiplied_feat_vec[c].append('{0}[{1}]={2}'.format(self.name, pos - c, feat))
        return multiplied_feat_vec


class Lexicon:
    """
    the Lexicon class generates so-called lexicon features
    an instance of Lexicon() should be initialized for each lexicon file
    """
    def __init__(self, input_file):
        self.phrase_list = set()
        self.end_parts = set()
        self.mid_parts = set()
        self.start_parts = set()
        for line in open(input_file, encoding='UTF-8'):
            phrase = line.strip()
            self.phrase_list.add(phrase)
            words = phrase.split()
            if len(words) > 1:
                self.end_parts.add(words[-1])
                self.start_parts.add(words[0])
                for w in words[1:-1]:
                    self.mid_parts.add(w)

    def _get_word_feats(self, word):
        word_feats = []
        if word in self.phrase_list:
            word_feats.append('lone')
        if word in self.end_parts:
            word_feats.append('end')
        if word in self.start_parts:
            word_feats.append('start')
        if word in self.mid_parts:
            word_feats.append('mid')
        return word_feats

    def lex_eval_sentence(self, sentence):
        return [self._get_word_feats(word) for word in sentence]
