#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-
"""
bigram.py contains the Bigram class which implements a simple bigram model
which can be built from observations of type (word1, word2). Bigram models are
built and used by HunTag
"""

import sys
import math
import operator
import pickle
from collections import defaultdict

from tools import sentenceIterator


# Bigram or Trigram transition model
class TransModel:
    def __init__(self, tagField=-1, smooth=0.000000000000001, boundarySymbol='S', lmw=1.0, order=3):
        self._trigramCount = defaultdict(int)
        self.trigramLogProb = {}
        self._labda3 = 0.0

        self._bigramCount = defaultdict(int)
        self.bigramLogProb = {}
        self._unigramCount = defaultdict(float)
        self.unigramLogProb = {}
        self._obsCount = 0
        self.updated = True
        self._labda1 = 0.0
        self._labda2 = 0.0
        self.tags = set()
        self.reset()

        self._tagField = tagField
        self._logSmooth = math.log(float(smooth))
        self._boundarySymbol = boundarySymbol
        self._languageModelWeight = float(lmw)
        self._order = int(order)
        if self._order == 2:
            self.viterbi = self.viterbiBigram
        elif self._order == 3:
            self.viterbi = self.viterbiBigram
        else:
            print('Error: Transition modell order should be 2 or 3 got {0}!'.format(order), file=sys.stderr, flush=True)
            sys.exit(1)

        self._updateWarning = 'WARNING: Probabilities have not been recalculated since last input!'

    def reset(self):
        self._trigramCount = defaultdict(int)
        self.trigramLogProb = {}
        self._labda3 = 0.0

        self._bigramCount = defaultdict(int)
        self.bigramLogProb = {}
        self._unigramCount = defaultdict(float)
        self.unigramLogProb = {}
        self._obsCount = 0
        self._labda1 = 0.0
        self._labda2 = 0.0
        self.tags = set()

        self.updated = True

    # Tag a sentence given the probability dists. of words
    def tagSent(self, tagProbsByPos):
        return self.viterbi(tagProbsByPos)[1]

    # Train a Stream
    def train(self, inputStream):
        for sen, _ in sentenceIterator(inputStream):
            self.obsSequence((tok[self._tagField] for tok in sen))

    # Train a Sentence (Either way we count trigrams, but later we will not use them)
    def obsSequence(self, tagSequence):
        lastBefore = self._boundarySymbol
        last = self._boundarySymbol
        for tag in tagSequence:
            self.obs(tag, last, lastBefore)
            lastBefore = last
            last = tag
        # XXX Maybe we should make explicit difference between sentence begin sentence end
        self.obs(self._boundarySymbol, last, lastBefore)

    # Train a Bigram or Trigram (Notice: the order is reversed to be able to add bigrams as well as trigrams)
    def obs(self, first, second, third=None):
        self._trigramCount[(first, second, third)] += 1
        self._bigramCount[(first, second)] += 1
        self._unigramCount[second] += 1
        self._obsCount += 1
        self.updated = False

    # Close model, and compute probabilities after (possibly incremental) training
    def count(self):
        self.tags = set()
        self.trigramLogProb = {}
        self.bigramLogProb = {}
        self.unigramLogProb = {}

        bigramJointLogProb = {}

        # Compute unigram probs: P(t_n) = C(t_n)/sum_i(C(t_i))
        for tag, count in self._unigramCount.items():
            # if tag != self._boundarySymbol:
            self.tags.add(tag)
            self.unigramLogProb[tag] = math.log(count) - math.log(self._obsCount)

        # Compute bigram probs (Conditional probability using joint probabilities):
        # Unigram prob: P(t_n-1) = C(t_n)/sum_i(C(t_i)) = self.unigramLogProb[tag]
        # Joint prob (bigram): P(t_n-1, t_n) = C(t_n-1, t_n)/C(t_n-1) = bigramJointLogProb
        # Conditional prob (bigram): P(t_n|t_n-1) = P(t_n-1, t_n)/P(t_n-1) =
        #     bigramJointLogProb(tag1,tag2) - self.unigramLogProb[tag1]
        for pair, count in self._bigramCount.items():  # log(Bigram / Unigram)
            bigramJointLogProb[pair] = math.log(count) - math.log(self._unigramCount[pair[0]])
            self.bigramLogProb[pair] = bigramJointLogProb[pair] - self.unigramLogProb[pair[0]]

        if self._order == 3:
            # Compute trigram probs (Conditional probability using joint probabilities):
            # Joint prob (bigram): P(t_n-1, t_n) = C(t_n-1, t_n)/C(t_n-1) = bigramJointLogProb
            # Joint prob (trigram): P(t_n-2, t_n-1, t_n) = C(t_n-2, t_n-1, t_n)/C(t_n-2, t_n-1) = trigramJointLogProb
            # Conditional prob (trigram): P(t_n|t_n-2, t_n-1) = P(t_n-2, t_n-1, t_n)/P(t_n-2, t_n-1) =
            #     trigramJointLogProb(tag1, tag2, tag3) - bigramJointLogProb[tag1, tag2]
            for tri, count in self._trigramCount.items():  # log(Trigram / Bigram)
                trigramJointLogProb = math.log(count) - math.log(self._bigramCount[tri[0:2]])
                self.trigramLogProb[tri] = trigramJointLogProb - bigramJointLogProb[tri[0:2]]

        # Compute lambdas
        self._compute_lambda()

        self.updated = True

    def _compute_lambda(self):
        """
        This function originates from NLTK
        creates lambda values based upon training data

        NOTE: no need to explicitly reference C,
        it is contained within the tag variable :: tag == (tag,C)

        for each tag trigram (t1, t2, t3)
        depending on the maximum value of
        - f(t1,t2,t3)-1 / f(t1,t2)-1
        - f(t2,t3)-1 / f(t2)-1
        - f(t3)-1 / N-1

        increment l3,l2, or l1 by f(t1,t2,t3)

        ISSUES -- Resolutions:
        if 2 values are equal, increment both lambda values
        by (f(t1,t2,t3) / 2)
        """

        # temporary lambda variables
        tl1 = 0.0
        tl2 = 0.0
        tl3 = 0.0

        # for each t1,t2 in system
        for h1, h2, _ in self._trigramCount.keys():
            history = (h1, h2)

            # for each t3 given t1,t2 in system
            for tag in (t3 for t1, t2, t3 in self._trigramCount.keys() if t1 == h1 and t2 == h2):

                # if there has only been 1 occurrence of this tag in the data
                # then ignore this trigram.
                if self._unigramCount[tag] > 1:

                    # safe_div provides a safe floating point division
                    # it returns -1 if the denominator is 0
                    if self._order == 3:
                        c3 = self._safe_div((self._trigramCount[(h1, h2, tag)]-1), (self._bigramCount[history]-1))
                    else:
                        c3 = -2.0  # Never will be maximum
                    c2 = self._safe_div((self._bigramCount[(h2, tag)]-1), (self._unigramCount[h2]-1))
                    c1 = self._safe_div((self._unigramCount[tag]-1), (self._obsCount-1))


                    # if c1 is the maximum value:
                    if (c1 > c3) and (c1 > c2):
                        tl1 += self._trigramCount[(h1, h2, tag)]

                    # if c2 is the maximum value
                    elif (c2 > c3) and (c2 > c1):
                        tl2 += self._trigramCount[(h1, h2, tag)]

                    # if c3 is the maximum value
                    elif (c3 > c2) and (c3 > c1):
                        tl3 += self._trigramCount[(h1, h2, tag)]

                    # if c3, and c2 are equal and larger than c1
                    elif (c3 == c2) and (c3 > c1):
                        tl2 += float(self._trigramCount[(h1, h2, tag)]) /2.0
                        tl3 += float(self._trigramCount[(h1, h2, tag)]) /2.0

                    # if c1, and c2 are equal and larger than c3
                    # this might be a dumb thing to do....(not sure yet)
                    elif (c2 == c1) and (c1 > c3):
                        tl1 += float(self._trigramCount[(h1, h2, tag)]) /2.0
                        tl2 += float(self._trigramCount[(h1, h2, tag)]) /2.0

                    # otherwise there might be a problem
                    # eg: all values = 0
                    #else:
                        #print('Problem', c1, c2 ,c3)
                    #    pass

        # Lambda normalisation:
        # ensures that l1+l2+l3 = 1
        self._lambda1 = tl1 / (tl1+tl2+tl3)
        self._lambda2 = tl2 / (tl1+tl2+tl3)
        self._lambda3 = tl3 / (tl1+tl2+tl3)
        print('lambda1: {0}'.format(self._lambda1), file=sys.stderr, flush=True)
        print('lambda2: {0}'.format(self._lambda2), file=sys.stderr, flush=True)
        print('lambda3: {0}'.format(self._lambda3), file=sys.stderr, flush=True)

    def _safe_div(self, v1, v2):
        """
        Safe floating point division function, does not allow division by 0
        returns -1 if the denominator is 0
        """
        if v2 == 0:
            return -1.0
        else:
            return float(v1) / float(v2)

    # Allways the last argument is the nth...
    def logProb(self, first, second=None, third=None):
        if not self.updated:
            print(self._updateWarning, file=sys.stderr, flush=True)

        uni = self._logSmooth
        bi = self._logSmooth
        tri = self._logSmooth
        if third is not None and (first, second, third) in self.trigramLogProb:
            return self.trigramLogProb[(first, second, third)]

        if second is not None and third is None and (first, second) in self.bigramLogProb:
            return self.bigramLogProb[(first, second)]

        if second is not None and (first, second) in self.bigramLogProb:
            bi = self.bigramLogProb[(first, second)]

        if first is not None and first in self.unigramLogProb:
            uni = self.unigramLogProb[first]

        return self._lambda1*uni + self._lambda2*bi + self._lambda3*tri

        print('Error in TransModel.logProb: got None for all arguments!', file=sys.stderr, flush=True)

    def prob(self, first, second=None, third=None):
        return math.exp(self.logProb(first, second, third))

    def writeToFile(self, fileName):
        self.tags.remove(self._boundarySymbol)
        with open(fileName, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def getModelFromFile(fileName):
        f = open(fileName, 'rb')
        model = pickle.load(f)
        f.close()
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
    def viterbiBigram(self, tagProbsByPos):
        # Make logprob from probs...
        tagProbsByPos = [dict([(key, math.log(val))
                               for key, val in probDist.items()])
                         for probDist in tagProbsByPos]
        V = [{}]
        path = {}
        states = self.tags

        if len(tagProbsByPos) == 1:
            label, prob = max(tagProbsByPos[0].items(), key=operator.itemgetter(1))
            return prob, [label]

        # Initialize base cases (t == 0)
        for y in states:
            V[0][y] = (self._languageModelWeight *
                          # We can come only from boundary symbols, so there is no need for loop and max...
                          # We must remember len(states) piece of states
                          self.logProb(self._boundarySymbol, y) +
                          tagProbsByPos[0][y])
            path[y] = [y]

        # Run Viterbi for t > 0
        for t in range(1, len(tagProbsByPos)):
            V.append({})
            newpath = {}
            # We extend the graph to every possible states
            # To every possible states, we can only come from the maximum
            # We remember this particular state
            for y in states:    # In t-1 we stand at y0 with some specific probability but we only could come from the maximum
                (prob, state) = max([(V[t - 1][y0] +
                                      self._languageModelWeight *
                                      # If we come from (z, y0)
                                      self.logProb(y0, y) +
                                      # If we extend, we get this
                                      tagProbsByPos[t][y],
                                      # This is the history, that we check
                                      # We compute max by y0
                                      y0) for y0 in states])
                # Now we stand at y, because it has the maximal probability 'prob'
                V[t][y] = prob
                # Extending the path with y, we came from state 'state'
                newpath[y] = path[state] + [y]

            # Don't need to remember the old paths
            path = newpath

        # At the end of the text we do a multiplication with a transition to check 'If we were in the end, would we come this way or not?'...
        (prob, state) = max([(V[len(tagProbsByPos) - 1][y0] + self.logProb(y0, self._boundarySymbol), y0) for y0 in states])
        return prob, path[state]

    def viterbiTrigram(self, tagProbsByPos):
        # Make logprob from probs...
        tagProbsByPos = [dict([(key, math.log(val))
                               for key, val in probDist.items()])
                         for probDist in tagProbsByPos]
        V = [{}]
        path = {}
        states = self.tags

        # Initialize base cases (t == 0)
        for z in states:
            for y in states:
                V[0][z, y] = (self._languageModelWeight *
                              self.logProb(self._boundarySymbol, self._boundarySymbol, y) +
                              tagProbsByPos[0][y])
                path[z, y] = [y]

        if len(tagProbsByPos) == 1:
            (prob, state, state2) = max([(V[0][z,y] + self.logProb(y, self._boundarySymbol), z, y) for z in states for y in states])
            return prob, path[state, state2]

        # Run Viterbi for t == 1
        V.append({})
        newpath = {}

        for z in states:
            for y in states:
                (prob, state) = max([(V[0][y0, z] +
                                    self._languageModelWeight *
                                    self.logProb(self._boundarySymbol, z, y) +
                                    tagProbsByPos[1][y],
                                    y0) for y0 in states])
                V[1][z, y] = prob
                newpath[z, y] = path[state, z] + [y]

        # Don't need to remember the old paths
        path = newpath

        # Run Viterbi for t > 1
        for t in range(2, len(tagProbsByPos)):
            V.append({})
            newpath = {}

            for z in states:
                for y in states:
                    (prob, state) = max([(V[t - 1][y0, z] +
                                        self._languageModelWeight *
                                        self.logProb(y0, z, y) +
                                        tagProbsByPos[t][y],
                                        y0) for y0 in states])
                    V[t][z, y] = prob
                    newpath[z, y] = path[state, z] + [y]

            # Don't need to remember the old paths
            path = newpath

        (prob, state, state2) = max([(V[len(tagProbsByPos) - 1][z,y] + self.logProb(y, self._boundarySymbol), z, y) for z in states for y in states])
        return prob, path[state, state2]

