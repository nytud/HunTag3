#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-
# Miscellaneous tools for HunTag

from operator import itemgetter
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

def index(arr, start, elem):
    ind = -1
    for i, e in enumerate(arr[start:]):
        if e <= elem:
            ind = start + i
        else:
            break
    return ind

# Keeps Feature/Label-Number translation maps, for faster computations
class BookKeeper():
    def __init__(self):
        self._counter = defaultdict(int)
        self._nameToNo = {}
        self.noToName = {}  # This is built only uppon reading back from file
        self._nextNo = 0
        self.numNotFound = self._nextNo - 1

    def numOfNames(self):
        return len(self._nameToNo)

    def makenoToName(self):
        self.noToName = {v: k for k, v in self._nameToNo.items()}

    def cutoff(self, cutoff):
        toDelete = set()
        for name, count in self._counter.items():
            if count < cutoff:
                toDelete.add(self._nameToNo[name])
                self._nameToNo.pop(name)
        newNameNo = dict(((name, i) for i, (name, no) in
                          enumerate(sorted(self._nameToNo.items(), key=itemgetter(1)))))
        del self._counter
        del self._nameToNo
        self._nameToNo = newNameNo
        return toDelete

    def getNoTag(self, name):
        if not name in self._nameToNo:
            return self.numNotFound
        return self._nameToNo[name]

    def getNoTrain(self, name):
        self._counter[name] += 1
        if not name in self._nameToNo:
            self._nameToNo[name] = self._nextNo
            self._nextNo += 1
        return self._nameToNo[name]

    def saveToFile(self, fileName):
        f = open(fileName, 'w', encoding='UTF-8')
        for name, no in sorted(self._nameToNo.items(), key=itemgetter(1)):
            f.write('{0}\t{1}\n'.format(name, str(no)))
        f.close()

    def readFromFile(self, fileName):
        self._nameToNo = {}
        self.noToName = {}
        for line in open(fileName, encoding='UTF-8'):
            l = line.strip().split()
            name, no = l[0], int(l[1])
            self._nameToNo[name] = no
            self.noToName[no] = name
            # This isn't used currently
            self._nextNo = no + 1
