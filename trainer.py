#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

"""
trainer.py is a module of HunTag and is used to train maxent models
"""

import sys
from operator import itemgetter
from collections import Counter, defaultdict
from sklearn.externals import joblib
from scipy.sparse import csr_matrix
import numpy as np
from array import array
from sklearn.linear_model import LogisticRegression

from tools import BookKeeper, sentenceIterator, featurizeSentence


class Trainer:
    def __init__(self, features, options):

        # Set clasifier algorithm here
        parameters = dict()
        self._model = LogisticRegression(**parameters)

        self._dataSizes = options['dataSizes']
        self._tagField = options['tagField']
        self._modelFileName = options['modelFileName']
        self._parameters = options['trainParams']
        self._cutoff = options['cutoff']
        self._featCounterFileName = options['featCounterFileName']
        self._labelCounterFileName = options['labelCounterFileName']
        self._features = features

        self._tokCount = -1  # Index starts from 0

        self._rows = array(self._dataSizes['rows'])
        self._cols = array(self._dataSizes['cols'])
        self._data = array(self._dataSizes['data'])
        self._labels = array(self._dataSizes['labels'])
        self._sentEnd = array(self._dataSizes['sentEnd'])  # Keep track of sentence boundaries
        self._matrix = None

        self._featCounter = BookKeeper()
        self._labelCounter = BookKeeper()
        self._usedFeats = None
        if 'usedFeats' in options and options['usedFeats']:
            self._usedFeats = set([line.strip()
                                  for line in open(options['usedFeats'], encoding='UTF-8')])

    def save(self):
        print('saving model...', end='', file=sys.stderr, flush=True)
        joblib.dump(self._model, '{0}'.format(self._modelFileName))
        print('done\nsaving feature and label lists...', end='', file=sys.stderr, flush=True)
        self._featCounter.saveToFile(self._featCounterFileName)
        self._labelCounter.saveToFile(self._labelCounterFileName)
        print('done', file=sys.stderr, flush=True)

    def _index(self, arr, start, elem):
        ind = -1
        for i, e in enumerate(arr[start:]):
            if e <= elem:
                ind = start + i
            else:
                break
        return ind

    def _updateSentEnd(self, sentEnds, rowNums):
        newEnds = array(self._dataSizes['sentEnd'])
        vbeg = 0
        for end in sentEnds:
            vend = self._index(rowNums, vbeg, end)
            if vend > 0:
                newEnds.append(vend)
                vbeg = vend + 1
        return newEnds

    def _convertToNPArray(self):
        rowsNP = np.array(self._rows, dtype=self._dataSizes['rowsNP'])
        colsNP = np.array(self._cols, dtype=self._dataSizes['cols'])
        dataNP = np.array(self._data, dtype=self._dataSizes['data'])
        labelsNP = np.array(self._labels, dtype=self._dataSizes['labels'])
        del self._rows
        del self._cols
        del self._data
        del self._labels
        self._rows = rowsNP
        self._cols = colsNP
        self._data = dataNP
        self._labels = labelsNP

    def _makeSparseArray(self, rowNum, colNum):
        print('creating training problem...', end='', file=sys.stderr, flush=True)
        matrix = csr_matrix((self._data, (self._rows, self._cols)), shape=(rowNum, self._cols.max()+1),
                            dtype=self._dataSizes['data'])
        del self._rows
        del self._cols
        del self._data
        print('done!', file=sys.stderr, flush=True)
        return matrix

    def cutoffFeats(self):
        self._convertToNPArray()
        colNum = self._featCounter.numOfNames()
        if self._cutoff < 2:
            self._matrix = self._makeSparseArray(self._tokCount, colNum)
        else:
            print('discarding features with less than {0} occurences...'.format(
                self._cutoff), end='', file=sys.stderr, flush=True)

            toDelete = self._featCounter.cutoff(self._cutoff)
            print('done!\nreducing training events by {0}...'.format(len(toDelete)),
                  end='', file=sys.stderr, flush=True)
            # ...that are not in featCounter anymore
            indicesToKeepNP = np.array([ind for ind, featNo in enumerate(self._cols)
                                        if featNo not in toDelete],
                                       dtype=self._dataSizes['cols'])
            del toDelete

            # Reduce cols
            colsNPNew = self._cols[indicesToKeepNP]
            del self._cols
            self._cols = colsNPNew

            # Reduce data
            dataNPNew = self._data[indicesToKeepNP]
            del self._data
            self._data = dataNPNew

            # Reduce rows
            rowsNPNew = self._rows[indicesToKeepNP]
            rowNumKeep = np.unique(rowsNPNew)
            rowNum = rowNumKeep.shape[0]
            colNum = indicesToKeepNP.shape[0]
            del self._rows
            self._rows = rowsNPNew
            del indicesToKeepNP

            # Reduce labels
            labelsNPNew = self._labels[rowNumKeep]
            del self._labels
            self._labels = labelsNPNew

            # Update sentence end markers
            newEnd = self._updateSentEnd(self._sentEnd, rowNumKeep)
            del self._sentEnd
            self._sentEnd = newEnd
            del rowNumKeep

            print('done!', file=sys.stderr, flush=True)
            matrix = self._makeSparseArray(rowNum, colNum)
            print('updating indices...', end='', file=sys.stderr, flush=True)

            # Update rowNos
            rows, _ = matrix.nonzero()
            matrixNew = matrix[np.unique(rows), :]
            del matrix
            del rows

            # Update featNos
            _, cols = matrixNew.nonzero()
            self._matrix = matrixNew[:, np.unique(cols)]
            del matrixNew
            del cols

            print('done!', file=sys.stderr, flush=True)

    # Input need featurizing
    def getEvents(self, data):
        print('featurizing sentences...', end='', file=sys.stderr, flush=True)
        senCount = 0
        tokIndex = -1  # Index starts from 0
        for sen, _ in sentenceIterator(data):
            senCount += 1
            sentenceFeats = featurizeSentence(sen, self._features)
            for c, tok in enumerate(sen):
                tokIndex += 1
                tokFeats = sentenceFeats[c]
                if self._usedFeats:
                    tokFeats = [feat for feat in tokFeats
                                if feat in self._usedFeats]
                self._addContext(tokFeats, tok[self._tagField], tokIndex)
            self._sentEnd.append(tokIndex)
            if senCount % 1000 == 0:
                print('{0}...'.format(str(senCount)), end='', file=sys.stderr, flush=True)

        self._tokCount = tokIndex + 1
        print('{0}...done!'.format(str(senCount)), file=sys.stderr, flush=True)

    # Already featurized input
    def getEventsFromFile(self, data):
        tokIndex = -1  # Index starts from 0
        for line in data:
            line = line.strip()
            if len(line) > 0:
                tokIndex += 1
                l = line.split()
                label, feats = l[0], l[1:]
                self._addContext(feats, label, tokIndex)
            self._sentEnd.append(tokIndex)
        self._tokCount = tokIndex + 1

    def _addContext(self, tokFeats, label, curTok):
        rowsAppend = self._rows.append
        colsAppend = self._cols.append
        dataAppend = self._data.append

        # features are sorted to ensure identical output
        # no matter where the features are coming from
        for featNumber in set([self._featCounter.getNoTrain(feat)
                               for feat in sorted(tokFeats)]):
            rowsAppend(curTok)
            colsAppend(featNumber)
            dataAppend(1)

        self._labels.append(self._labelCounter.getNoTrain(label))

    def mostInformativeFeatures(self):
        self._featCounter.makenoToName()
        self._labelCounter.makenoToName()
        featnoToName = self._featCounter.noToName
        labelnoToName = self._labelCounter.noToName
        rows = self._rows
        cols = self._cols
        labels = self._labels
        featSorted = defaultdict(Counter)
        for ind, rowNum in enumerate(rows):
            featSorted[cols[ind]][labels[rowNum]] += 1

        ranking = []
        for featNum, occurences in sorted(featSorted.items()):
            sumOccurences = 0
            labelCounts = []
            for label, count in sorted(occurences.items(), key=itemgetter(1), reverse=True):
                sumOccurences += count
                labelCounts.append((labelnoToName[label], count))

            maximum = labelCounts[0][1] / sumOccurences  # Because it's sorted reverse
            labelCountsProb = []
            for label, count in occurences.items():
                labelCountsProb.append((labelnoToName[label], count / sumOccurences))
            featData = '{0}\t{1}\t{2}\t{3}'.format(featnoToName[featNum], sumOccurences,
                                                   '/'.join(['{0}:{1}'.format(l, c) for l, c in labelCounts]),
                                                   '/'.join(['{0}:{1}'.format(l, p) for l, p in labelCountsProb]))
            ranking.append((maximum, sumOccurences, featData))
        for _, _, text in sorted(ranking, reverse=True):
            print(text)

    def toCRFsuite(self):
        self._featCounter.makenoToName()
        self._labelCounter.makenoToName()
        featnoToName = self._featCounter.noToName
        labelnoToName = self._labelCounter.noToName
        sentEnd = self._sentEnd
        matrix = self._matrix.tocsr()
        labels = self._labels
        beg = 0
        for end in sentEnd:
            for row in range(beg, end + 1):
                columns = [featnoToName[col].replace(':', 'colon') for col in matrix[row, :].nonzero()[1]]
                print('{0}\t{1}'.format(labelnoToName[labels[row]], '\t'.join(columns)))
            print()  # Sentence separator blank line
            beg = end + 1

    def train(self):
        print('training with option(s) "{0}"...'.format(
            self._parameters), end='', file=sys.stderr, flush=True)
        _ = self._model.fit(self._matrix, self._labels)
        print('done', file=sys.stderr, flush=True)
