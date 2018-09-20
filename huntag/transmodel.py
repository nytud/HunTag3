#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-
"""
bigram.py contains the Bigram class which implements a simple bigram model
which can be built from observations of type (word1, word2). Bigram models are
built and used by HunTag
"""

import sys
import math
import pickle
from collections import Counter

from huntag.tools import sentence_iterator


def safe_div(v1, v2):
    """
    Safe floating point division function, does not allow division by 0
    returns -1 if the denominator is 0

    Args:
        v1: numerator
        v2: denominator
    """
    if v2 == 0:
        return -1.0
    else:
        return float(v1) / float(v2)


# Bigram or Trigram transition model
class TransModel:
    def __init__(self, tag_field=-1, smooth=0.000000000000001, boundary_symbol='S', lmw=1.0, order=3):
        self._unigram_count = Counter()
        self.unigram_logprob = {}
        self._lambda1 = 0.0
        self._bigram_count = Counter()
        self.bigram_logprob = {}
        self._lambda2 = 0.0
        self._trigram_count = Counter()
        self.trigram_logprob = {}
        self._lambda3 = 0.0
        self._obs_count = 0
        self._sent_count = 0
        self.tags = set()
        self.updated = True
        self.reset()

        self._tag_field = tag_field
        self._log_smooth = math.log(float(smooth))
        self._boundary_symbol = boundary_symbol
        self._language_model_weight = float(lmw)
        self._order = int(order)
        if self._order == 2:
            self.viterbi = self._viterbi_bigram
        elif self._order == 3:
            self.viterbi = self._viterbi_trigram
        else:
            print('Error: Transition modell order should be 2 or 3 got {0}!'.format(order), file=sys.stderr, flush=True)
            sys.exit(1)

        self._update_warning = 'WARNING: Probabilities have not been recalculated since last input!'

    def reset(self):
        self._unigram_count = Counter()
        self._bigram_count = Counter()
        self._trigram_count = Counter()
        self._obs_count = 0
        self._sent_count = 0
        self.updated = True

    # Tag a sentence given the probability dists. of words
    def tag_sent(self, tagprobs_by_pos):
        return self.viterbi(tagprobs_by_pos)[1]

    # Train a Stream
    def train(self, input_stream):
        for sen, _ in sentence_iterator(input_stream):
            self.obs_sequence((tok[self._tag_field] for tok in sen))

    # Train a Sentence (Either way we count trigrams, but later we will not use them)
    def obs_sequence(self, tag_sequence):
        last_before = self._boundary_symbol
        last = self._boundary_symbol
        # Add the two boundary symbol to the counts...
        self._bigram_count[self._boundary_symbol, self._boundary_symbol] += 1
        self._unigram_count[self._boundary_symbol] += 2
        self._obs_count += 2
        # Count sentences, for later normalization
        self._sent_count += 1
        for tag in tag_sequence:
            self.obs(last_before, last, tag)
            last_before = last
            last = tag
        # XXX Maybe we should make explicit difference between sentence begin sentence end
        self.obs(last_before, last, self._boundary_symbol)

    # Train a Bigram or Trigram (Compute trigrams, and later optionally use bigrams only)
    # To train directly a bigram use: obs(nMinusOne=firstToken, nth=secondToken) or obs(None, firstToken, secondToken)
    def obs(self, n_minus_two=None, n_minus_one=None, nth=None):
        self._trigram_count[n_minus_two, n_minus_one, nth] += 1
        self._bigram_count[n_minus_one, nth] += 1
        self._unigram_count[nth] += 1
        self._obs_count += 1
        self.updated = False

    # Close model, and compute probabilities after (possibly incremental) training
    def compile(self):
        self.trigram_logprob = {}
        self.bigram_logprob = {}
        self.unigram_logprob = {}

        bigram_joint_logprob = {}
        
        if self._order == 2:
            # Remove (self._boundary_symbol, self._boundary_symbol) as it has no meaning for bigrams...
            self._bigram_count.pop((self._boundary_symbol, self._boundary_symbol), None)
            # Remove the Unigram count of the removed self._boundary_symbol
            self._unigram_count[self._boundary_symbol] -= self._sent_count
            self._obs_count -= self._sent_count
            # Reset, as incremental training (if there is any) will start from here...
            self._sent_count = 0

        # Compute unigram probs: P(t_n) = C(t_n)/sum_i(C(t_i))
        self.tags = set(self._unigram_count.keys())
        self.unigram_logprob = {tag: math.log(count) - math.log(self._obs_count)
                                for tag, count in self._unigram_count.items()}

        # Compute bigram probs (Conditional probability using joint probabilities):
        # Unigram prob: P(t_n-1) = C(t_n)/sum_i(C(t_i)) = self.unigram_logprob[tag]
        # Joint prob (bigram): P(t_n-1, t_n) = C(t_n-1, t_n)/C(t_n-1) = bigram_joint_logprob(tag1,tag2)
        # Conditional prob (bigram): P(t_n|t_n-1) = P(t_n-1, t_n)/P(t_n-1) =
        #     bigram_joint_logprob(tag1,tag2) - self.unigram_logprob[tag1]
        for pair, count in self._bigram_count.items():  # log(Bigram / Unigram)
            bigram_joint_logprob[pair] = math.log(count) - math.log(self._unigram_count[pair[0]])
            self.bigram_logprob[pair] = bigram_joint_logprob[pair] - self.unigram_logprob[pair[0]]

        if self._order == 3:
            # Compute trigram probs (Conditional probability using joint probabilities):
            # Joint prob (bigram): P(t_n-1, t_n) = C(t_n-1, t_n)/C(t_n-1) = bigram_joint_logprob(tag1,tag2)
            # Joint prob (trigram): P(t_n-2, t_n-1, t_n) = C(t_n-2, t_n-1, t_n)/C(t_n-2, t_n-1) =
            #     trigram_joint_logprob(tag1, tag2, tag3)
            # Conditional prob (trigram): P(t_n|t_n-2, t_n-1) = P(t_n-2, t_n-1, t_n)/P(t_n-2, t_n-1) =
            #     trigram_joint_logprob(tag1, tag2, tag3) - bigram_joint_logprob(tag1, tag2)
            for tri, count in self._trigram_count.items():  # log(Trigram / Bigram)
                trigram_joint_logprob = math.log(count) - math.log(self._bigram_count[tri[0:2]])
                self.trigram_logprob[tri] = trigram_joint_logprob - bigram_joint_logprob[tri[0:2]]

        # Compute lambdas
        self._compute_lambda()

        self.updated = True

    def _compute_lambda(self):
        """
        This function originates from NLTK
        creates lambda values based upon training data (Brants 2000, Figure 1)

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

        # for each t3 given t1,t2 in system
        for h1, h2, tag in self._trigram_count.keys():

            # if there has only been 1 occurrence of this tag in the data
            # then ignore this trigram.
            if self._unigram_count[tag] > 1:

                # safe_div provides a safe floating point division
                # it returns -1 if the denominator is 0
                if self._order == 3:
                    c3 = safe_div(self._trigram_count[h1, h2, tag] - 1, self._bigram_count[h1, h2] - 1)
                else:
                    c3 = -2.0  # Never will be maximum
                c2 = safe_div(self._bigram_count[h2, tag] - 1, self._unigram_count[h2] - 1)
                c1 = safe_div(self._unigram_count[tag] - 1, self._obs_count - 1)

                # if c1 is the maximum value:
                if (c1 > c3) and (c1 > c2):
                    tl1 += self._trigram_count[h1, h2, tag]

                # if c2 is the maximum value
                elif (c2 > c3) and (c2 > c1):
                    tl2 += self._trigram_count[h1, h2, tag]

                # if c3 is the maximum value
                elif (c3 > c2) and (c3 > c1):
                    tl3 += self._trigram_count[h1, h2, tag]

                # if c3, and c2 are equal and larger than c1
                elif (c3 == c2) and (c3 > c1):
                    tl2 += self._trigram_count[h1, h2, tag] / 2.0
                    tl3 += self._trigram_count[h1, h2, tag] / 2.0

                # if c1, and c2 are equal and larger than c3
                # this might be a dumb thing to do....(not sure yet)
                elif (c2 == c1) and (c1 > c3):
                    tl1 += self._trigram_count[h1, h2, tag] / 2.0
                    tl2 += self._trigram_count[h1, h2, tag] / 2.0

                """
                # otherwise there might be a problem
                # eg: all values = 0
                else:
                    print('Problem', c1, c2 ,c3)
                    pass
                """
        # Lambda normalisation:
        # ensures that l1+l2+l3 = 1
        self._lambda1 = tl1 / (tl1 + tl2 + tl3)
        self._lambda2 = tl2 / (tl1 + tl2 + tl3)
        self._lambda3 = tl3 / (tl1 + tl2 + tl3)
        print('lambda1: {0}\nlambda2: {1}\nlambda3: {2}'.format(self._lambda1, self._lambda2, self._lambda3),
              file=sys.stderr, flush=True)

    # Allways use smoothing with lambdas (Brants 2000, formula 2-6)
    def _log_prob(self, n_minus_two=None, n_minus_one=None, nth=None):
        if not self.updated:
            print(self._update_warning, file=sys.stderr, flush=True)

        # Trigram, which is seen in training set or using smoothing
        tri = self.trigram_logprob.get((n_minus_two, n_minus_one, nth), self._log_smooth)

        # Bigram, which is seen in training set or using smoothing
        bi = self.bigram_logprob.get((n_minus_one, nth), self._log_smooth)

        # Unigram, which is seen in training set or using smoothing
        uni = self.unigram_logprob.get(nth, self._log_smooth)

        # Weighted by lambdas...
        return self._lambda1 * uni + self._lambda2 * bi + self._lambda3 * tri

    def prob(self, n_minus_two=None, n_minus_one=None, nth=None):
        return math.exp(self._log_prob(n_minus_two, n_minus_one, nth))

    def save_to_file(self, file_name):
        self.tags.remove(self._boundary_symbol)
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)

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
    def _viterbi_bigram(self, tagprobs_by_pos):
        # Make logprob from probs...
        tagprobs_by_pos = [dict([(key, math.log(val))
                                 for key, val in probDist.items()])
                           for probDist in tagprobs_by_pos]
        v = [{}]
        path = {}
        states = self.tags

        # Initialize base cases (t == 0)
        for y in states:
            v[0][y] = (self._language_model_weight *
                       # We can come only from boundary symbols, so there is no need for loop and max...
                       # We must remember len(states) piece of states
                       self._log_prob(None, self._boundary_symbol, y) +
                       tagprobs_by_pos[0][y])
            path[y] = [y]

        # Run Viterbi for t > 0
        for t in range(1, len(tagprobs_by_pos)):
            v.append({})
            newpath = {}
            # We extend the graph to every possible states
            # To every possible states, we can only come from the maximum
            # We remember this particular state
            for y in states:
                # In t-1 we stand at y0 with some specific probability but we only could come from the maximum
                (prob, state) = max([(v[t - 1][y0] +
                                      self._language_model_weight *
                                      # If we come from (z, y0)
                                      self._log_prob(None, y0, y) +
                                      # If we extend, we get this
                                      tagprobs_by_pos[t][y],
                                      # This is the history, that we check
                                      # We compute max by y0
                                      y0) for y0 in states])
                # Now we stand at y, because it has the maximal probability 'prob'
                v[t][y] = prob
                # Extending the path with y, we came from state 'state'
                newpath[y] = path[state] + [y]

            # Don't need to remember the old paths
            path = newpath

        # At the end of the text we do a multiplication with a transition to check
        # 'If we were in the end, would we come this way or not?'...
        (prob, state) = max([(v[len(tagprobs_by_pos) - 1][y] + self._log_prob(None, y, self._boundary_symbol), y)
                             for y in states])
        return prob, path[state]

    def _viterbi_trigram(self, tag_probs_by_pos):
        # Make logprob from probs...
        tag_probs_by_pos = [dict([(key, math.log(val))
                                  for key, val in probDist.items()])
                            for probDist in tag_probs_by_pos]
        v = [{}]
        path = {}
        states = self.tags

        # Initialize base cases (t == 0)
        for z in states:
            for y in states:
                v[0][z, y] = (self._language_model_weight *
                              self._log_prob(self._boundary_symbol, self._boundary_symbol, y) +
                              tag_probs_by_pos[0][y])
                path[z, y] = [y]

        if len(tag_probs_by_pos) > 1:
            # Run Viterbi for t == 1
            v.append({})
            newpath = {}

            for z in states:
                for y in states:
                    (prob, state) = max([(v[0][y0, z] +
                                          self._language_model_weight *
                                          self._log_prob(self._boundary_symbol, z, y) +
                                          tag_probs_by_pos[1][y],
                                          y0) for y0 in states])
                    v[1][z, y] = prob
                    newpath[z, y] = path[state, z] + [y]

            # Don't need to remember the old paths
            path = newpath

            # Run Viterbi for t > 1
            for t in range(2, len(tag_probs_by_pos)):
                v.append({})
                newpath = {}

                for z in states:
                    for y in states:
                        (prob, state) = max([(v[t - 1][y0, z] +
                                              self._language_model_weight *
                                              self._log_prob(y0, z, y) +
                                              tag_probs_by_pos[t][y],
                                              y0) for y0 in states])
                        v[t][z, y] = prob
                        newpath[z, y] = path[state, z] + [y]

                # Don't need to remember the old paths
                path = newpath

        # Micro-optimalization: Brants (2000) say self._log_prob(None, y, self._boundary_symbol),
        # but why not self._log_prob(z, y, self._boundary_symbol) ?
        (prob, state, state2) = max([(v[len(tag_probs_by_pos) - 1][z, y] +
                                      # self._log_prob(z, y, self._boundary_symbol), z, y)
                                      self._log_prob(None, y, self._boundary_symbol), z, y)
                                     for z in states for y in states])
        return prob, path[state, state2]
