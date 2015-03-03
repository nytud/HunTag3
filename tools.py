#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-
# Miscellaneous tools for HunTag
from collections import defaultdict
import sys


def sentenceIterator(inputStream):
    currSen = []
    currComment = None
    for line in inputStream:
        line = line.strip()
        # Comment handling
        if line.startswith('"""'):
            if len(currSen) == 0:  # Comment before sentence
                currComment = line
            else:  # Error: Comment in the middle of sentence
                print('ERROR: comments are only allowed before a sentence!',
                      file=sys.stderr, flush=True)
                sys.exit(1)
        # Blank line handling
        elif len(line) == 0:
            if currSen:  # End of sentence
                yield currSen, currComment
                currSen = []
                currComment = None
            else:  # Error: Multiple blank line
                print('ERROR: wrong formatted sentences, only one blank line allowed!',
                      file=sys.stderr, flush=True)
                sys.exit(1)
        else:
            currSen.append(line.split())
    # XXX Here should be an error because of missing blank line before EOF
    if currSen:
        yield currSen, currComment


def writeSentence(sen, out=sys.stdout, comment=None):
    if comment:
        out.write('{0}\n'.format(comment))
    for tok in sen:
        out.write('{0}\n'.format('\t'.join(tok)))
    out.write('\n')


def featurizeSentence(sen, features):
    sentenceFeats = [[] for _ in sen]
    for feature in features.values():
        for c, feats in enumerate(feature.evalSentence(sen)):
            sentenceFeats[c] += feats
    return sentenceFeats


# Keeps Feature-Number translation maps, for faster computations
class BookKeeper():
    def __init__(self):
        self._featCounter = defaultdict(int)
        self._featToNo = {}
        self.noToFeat = {}
        # XXX Liblinear segfaults if indices start from 0
        self._nextFeatNo = 1

    def cutoff(self, cutoff):
        toDelete = set()
        for feat, count in self._featCounter.items():
            if count < cutoff:
                toDelete.add(feat)
        for feat in toDelete:
            self._featCounter.pop(feat)
            self.noToFeat.pop(self._featToNo[feat])
            self._featToNo.pop(feat)

    def getNo(self, feat):
        self._featCounter[feat] += 1
        if not feat in self._featToNo:
            self._featToNo[feat] = self._nextFeatNo
            self.noToFeat[self._nextFeatNo] = feat
            self._nextFeatNo += 1
        return self._featToNo[feat]

    def saveToFile(self, fileName):
        f = open(fileName, 'w', encoding='UTF-8')
        for feat, no in self._featToNo.items():
            f.write('{0}\t{1}\n'.format(feat, str(no)))
        f.close()

    def readFromFile(self, fileName):
        self._featToNo = {}
        self.noToFeat = {}
        for line in open(fileName, encoding='UTF-8'):
            l = line.strip().split()
            # Feats not sorted by their numbers!
            feat, no = l[0], int(l[1])
            self._featToNo[feat] = no
            self.noToFeat[no] = feat
            # XXX This is wrong and not used currently
            self._nextFeatNo = no + 1
