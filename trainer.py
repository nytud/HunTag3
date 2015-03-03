#!/usr/bin/python
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

"""
trainer.py is a module of HunTag and is used to train maxent models
"""

import sys
from ctypes import c_int

from tools import BookKeeper, sentenceIterator, featurizeSentence
from liblinearutil import train, problem, parameter, save_model


class Trainer():
    def __init__(self, features, options):
        self.model = None
        self.tagField = options['tagField']
        self.modelFileName = options['modelFileName']
        self.parameters = options['trainParams']
        self.cutoff = options['cutoff']
        self.featCounterFileName = options['featCounterFileName']
        self.labelCounterFileName = options['labelCounterFileName']
        self.features = features
        self.labels = []
        self.contexts = []
        self.labelCounter = BookKeeper()
        self.featCounter = BookKeeper()
        self.usedFeats = None
        if options['usedFeats']:
            self.usedFeats = set([line.strip()
                                  for line in options['usedFeats']])

    def save(self):
        sys.stderr.write('saving model...')
        save_model(self.modelFileName, self.model)
        sys.stderr.write('done\nsaving feature and label lists...')
        self.featCounter.saveToFile(self.featCounterFileName)
        self.labelCounter.saveToFile(self.labelCounterFileName)
        sys.stderr.write('done\n')

    def cutoffFeats(self):
        if self.cutoff >= 2:
            sys.stderr.write('discarding features with\
            less than {0} occurences...'.format(self.cutoff))
            self.featCounter.cutoff(self.cutoff)
            sys.stderr.write('done!\nreducing training events...')
            # ...that are not in featCounter anymore
            self.contexts = [dict([(number, value)
                                  for number, value in context.iteritems()
                                  if number in self.featCounter.noToFeat])
                             for context in self.contexts]
            sys.stderr.write('done!\n')

    def getEvents(self, data, out_file_name=None):
        sys.stderr.write('featurizing sentences...')
        if out_file_name:
            out_file = open(out_file_name, 'w')
        senCount = 0
        for sen, _ in sentenceIterator(data):
            senCount += 1
            sentenceFeats = featurizeSentence(sen, self.features)
            for c, tok in enumerate(sen):
                tokFeats = sentenceFeats[c]
                if self.usedFeats:
                    tokFeats = [feat for feat in tokFeats
                                if feat in self.usedFeats]
                if out_file_name:
                    out_file.write('{0}\t{1}\n'.format(tok[self.tagField],
                                                       ' '.join(tokFeats)))
                self.addContext(tokFeats, tok[self.tagField])
            if out_file_name:
                out_file.write('\n')
            if senCount % 1000 == 0:
                sys.stderr.write('{0}...'.format(str(senCount)))

        sys.stderr.write('{0}...done!\n'.format(str(senCount)))

    def getEventsFromFile(self, fileName):
        for line in open(fileName):
            line = line.strip()
            if len(line) > 0:
                l = line.split()
                label, feats = l[0], l[1:]
                self.addContext(feats, label)

    def addContext(self, tokFeats, label):
        """
        features are sorted to ensure identical output
        no matter where the features are coming from
        """
        tokFeats.sort()
        featNumbers = set([self.featCounter.getNo(feat)
                           for feat in tokFeats])

        # This is liblinear dependent
        # Need the extra parentheses to distinct array and array object
        context = ((c_int * 2) * len(featNumbers))()
        for i, no in enumerate(featNumbers):
            context[i][0] = no
            context[i][1] = 1
        labelNumber = self.labelCounter.getNo(label)
        self.contexts.append(context)
        self.labels.append(labelNumber)

    def train(self):
        sys.stderr.write('creating training problem...')
        prob = problem(self.labels, self.contexts)
        sys.stderr.write('done\ntraining with option(s) "{0}"...'.format(
            self.parameters))
        self.model = train(prob, parameter(self.parameters))
        sys.stderr.write('done\n')
