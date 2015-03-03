#!/usr/bin/python
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import math
import sys
import os

from bigram import Bigram, viterbi
from liblinearutil import load_model, predict
from tools import sentenceIterator, featurizeSentence


class Tagger():
    def __init__(self, features, options):
        self.features = features
        self.params = '-b 1'  # Order Liblinear to predict probability estimates
        self.lmw = options['lmw']
        sys.stderr.write('loading transition model...')
        self.transProbs = Bigram.getModelFromFile(options['bigramModelFileName'])
        sys.stderr.write('done\nloading observation model...')
        self.model = load_model(options['modelFileName'])
        self.labelCounter = options['labelCounter']
        self.featCounter = options['featCounter']
        sys.stderr.write('done\n')

    def tag_features(self, file_name):
        sen_feats = []
        senCount = 0
        for line in open(file_name):
            line = line.strip()
            if len(line) == 0:
                senCount += 1
                tagging = self.tag_sen_feats(sen_feats)
                yield [[tag] for tag in tagging]
                sen_feats = []
                if senCount % 1000 == 0:
                    sys.stderr.write('{0}...'.format(str(senCount)))
            sen_feats.append(line.split())
        sys.stderr.write('{0}...done\n'.format(str(senCount)))

    def tag_dir(self, dir_name):
        for fn in os.listdir(dir_name):
            sys.stderr.write('processing file {0}...'.format(fn))
            try:
                for sen, _ in self.tag_corp(open(os.path.join(dir_name, fn))):
                    yield sen, fn
            except:
                sys.stderr.write('error in file {0}\n'.format(fn))

    def tag_corp(self, inputStream):
        senCount = 0
        for sen, comment in sentenceIterator(inputStream):
            senCount += 1
            senFeats = featurizeSentence(sen, self.features)
            bestTagging = self.tag_sen_feats(senFeats)
            # Add tagging to sentence
            taggedSen = [tok + [bestTagging[c]] for c, tok in enumerate(sen)]
            yield taggedSen, comment
            if senCount % 1000 == 0:
                sys.stderr.write('{0}...'.format(str(senCount)))
        sys.stderr.write('{0}...done\n'.format(str(senCount)))

    def getLogTagProbsByPos(self, senFeats):
        # XXX We might add features, that are not in the training set
        # Get Sentence Features translated to numbers and contexts simultaneity
        contexts = [dict([(self.featCounter.getNo(feat), 1) for feat in feats])
                    for feats in senFeats]

        # This is liblinear dependent
        dummyOutcomes = [1 for _ in contexts]
        # pred_labels, ACC, pred_values
        _, _, probDistsByPos = predict(dummyOutcomes, contexts,
                                       self.model, self.params)

        # Apply math.log() to probDist's elements
        logTagProbsByPos = [dict([(self.labelCounter.noToFeat[i + 1],
                                   math.log(prob))
                                  for i, prob in enumerate(probDist)])
                            for probDist in probDistsByPos]

        return logTagProbsByPos

    def tag_sen_feats(self, sen_feats):
        logTagProbsByPos = self.getLogTagProbsByPos(sen_feats)
        _, bestTagging = viterbi(self.transProbs, logTagProbsByPos,
                                 self.lmw)
        return bestTagging
