#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-
"""
bigram.py contains the Bigram class which implements a simple bigram model
which can be built from observations of type (word1, word2). Bigram models are
built and used by HunTag
"""

import sys
import math
from collections import defaultdict


class Bigram:
    def __init__(self, smooth):
        self._bigramCount = defaultdict(int)
        self.bigramLogProb = {}
        self._unigramCount = defaultdict(float)
        self.unigramLogProb = {}
        self._obsCount = 0
        self.updated = True
        self.reset()
        self._boundarySymbol = 'S'
        self._smooth = smooth
        self._logSmooth = math.log(self._smooth)
        self._updateWarning = 'WARNING: Probabilities have not been \
                              recalculated since last input!'
        self.tags = set()

    def reset(self):
        self._bigramCount = defaultdict(int)
        self.bigramLogProb = {}
        self._unigramCount = defaultdict(float)
        self.unigramLogProb = {}
        self._obsCount = 0
        self.updated = True

    def obsSequence(self, tagSequence):
        last = self._boundarySymbol
        for tag in tagSequence:
            self.obs(last, tag)
            last = tag
        # XXX Maybe we should make explicit difference between sentence begin sentence end
        self.obs(last, self._boundarySymbol)

    def obs(self, first, second):
        self._bigramCount[(first, second)] += 1
        self._unigramCount[second] += 1
        self._obsCount += 1
        self.updated = False

    def count(self):
        self.tags = set()
        self.bigramLogProb = {}
        self.unigramLogProb = {}
        for pair, count in self._bigramCount.items():  # log(Bigram / Unigram)
            self.bigramLogProb[pair] = math.log(count) - math.log(self._unigramCount[pair[0]])
            # print(pair, count, self._unigramCount[first],
            #       count / self._unigramCount[first],
            #       math.log(count) - math.log(self._unigramCount[first]))

        for tag, count in self._unigramCount.items():
            # if tag != self._boundarySymbol:
            self.tags.add(tag)
            self.unigramLogProb[tag] = math.log(count / self._obsCount)

        self.updated = True

    def logProb(self, first, second):
        if not self.updated:
            print(self._updateWarning, file=sys.stderr, flush=True)

        if (first, second) in self.bigramLogProb:
            return self.bigramLogProb[(first, second)]
        else:
            return self._logSmooth

    def prob(self, first, second):
        return math.exp(self.logProb(first, second))

    def writeToFile(self, fileName):
        f = open(fileName, 'w', encoding='UTF-8')
        f.write('{0}\n'.format(str(self._smooth)))
        tagProbs = ['{0}:{1}'.format(tag, str(self.unigramLogProb[tag]))
                    for tag in self.tags if tag != self._boundarySymbol]
        f.write('{0}\n'.format(' '.join(tagProbs)))
        for t1 in self.tags:
            for t2 in self.tags:
                # It's better to make the format specifier explicit
                f.write('{0}\t{1}\t{2:.8f}\n'.format(t1, t2,
                                                     self.logProb(t1, t2)))
        f.close()

    @staticmethod
    def getModelFromFile(fileName):
        modelFile = open(fileName, encoding='UTF-8')
        smooth = float(modelFile.readline())
        model = Bigram(smooth)
        tagProbs = modelFile.readline().split()
        for tagAndProb in tagProbs:
            tag, prob = tagAndProb.split(':')
            model.tags.add(tag)
            model.unigramLogProb[tag] = float(prob)
        for line in modelFile:
            l = line.split()
            t1, t2, logProb = l[0], l[1], float(l[2])
            model.bigramLogProb[(t1, t2)] = logProb
        return model


"""
source: http://en.wikipedia.org/wiki/Viterbi_algorithm
The code has been modified to match our Bigram models:
- models are dictionaries with tuples as keys
- starting probabilities are not separate and end probabilities are also
taken into consideration
- transProbs should be a Bigram instance
- tagProbsByPos should be a list containing, for each position,
  the probability distribution over tags as returned by the maxent model
- all probabilities are expected to be in log space
"""


def viterbi(transProbs, tagProbsByPos, languageModelWeight,
            boundarySymbol='S'):
    V = [{}]
    path = {}
    states = transProbs.tags
    # Initialize base cases (t == 0)
    for y in states:
        trProb = transProbs.logProb(boundarySymbol, y)
        tagProb = tagProbsByPos[0][y]
        unigrProb = transProbs.unigramLogProb[y]
        V[0][y] = trProb + tagProb - unigrProb
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1, len(tagProbsByPos)):
        V.append({})
        newpath = {}

        for y in states:
            if t == len(tagProbsByPos):
                (prob, state) = max([languageModelWeight *
                                     transProbs.logProb(y, boundarySymbol) +
                                     (V[t - 1][y0] +
                                      languageModelWeight *
                                      transProbs.logProb(y0, y) +
                                      tagProbsByPos[t][y] -
                                      # dividing by a priori probability so as
                                      # not to count it twice
                                      transProbs.unigramLogProb[y],
                                      y0) for y0 in states])
            else:
                (prob, state) = max([(V[t - 1][y0] +
                                      languageModelWeight *
                                      transProbs.logProb(y0, y) +
                                      tagProbsByPos[t][y] -
                                      # dividing by a priori probability so as
                                      # not to count it twice
                                      languageModelWeight *
                                      transProbs.unigramLogProb[y],
                                      y0) for y0 in states])
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath

    (prob, state) = max([(V[len(tagProbsByPos) - 1][y], y) for y in states])
    return prob, path[state]
