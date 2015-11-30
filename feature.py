#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-
"""
feature.py is a module of HunTag. The Feature class is used for representing
a feature type and calculating its value for some input. Feature instances are
created by the getFeatureSet function in huntag.py.
"""

import sys

import features


class Feature:
    def __init__(self, kind, name, actionName, fields, radius, cutoff,
                 options):
        self.kind = kind
        self.name = name
        self.actionName = actionName
        self.fields = fields
        if self.kind == 'lex' and len(self.fields) != 1:
            print('Error: Feature "{0}" field count must be\
            one not {1}!'.format(self.name, self.fields), file=sys.stderr, flush=True)
            sys.exit(1)
        self.radius = int(radius)
        self.cutoff = int(cutoff)
        self.options = options
        if kind == 'lex':
            if len(self.options) > 0:
                print('Lexicon features do not yet support options',
                      file=sys.stderr, flush=True)
                sys.exit(1)
            self.lexicon = Lexicon(actionName)  # Load input file
        elif kind in ('token', 'sentence'):
            if actionName not in features.__dict__:
                print('Unknown operator named {0}\n'.format(actionName),
                      file=sys.stderr, flush=True)
                sys.exit(1)
            self.function = features.__dict__[actionName]
        else:
            print('Unknown kind named {0}'.format(kind), file=sys.stderr,
                  flush=True)
            sys.exit(1)

    def evalSentenceToken(self, sentence):
        featVec = []
        for word in sentence:
            # Pick the relevant fields (label can be not just the last field)
            fieldVec = [word[field] for field in self.fields]
            if len(self.options) > 0:
                # Options dict if there is any
                fieldVec.append(self.options)
            # Unpack argument list for function, then append the result
            featVec.append(self.function(*fieldVec))
        return featVec

    # Word will be substituted by its features from the Lexicon
    # self.fields denote the column of the word
    def evalSentenceLex(self, sentence):
        wordList = [word[self.fields[0]] for word in sentence]
        return self.lexicon.lexEvalSentence(wordList)

    def evalSentenceSentence(self, sentence):
        # XXX Should be better to make function signatures more strict
        if len(self.options) > 0:
            return self.function(sentence, self.fields, self.options)
        else:
            return self.function(sentence, self.fields)

    def evalSentence(self, sentence):
        if self.kind == 'token':
            featVec = self.evalSentenceToken(sentence)
        elif self.kind == 'lex':
            featVec = self.evalSentenceLex(sentence)
        elif self.kind == 'sentence':
            featVec = self.evalSentenceSentence(sentence)
        else:
            print('evalSentence: Unknown kind named {0}'.format(self.kind),
                  file=sys.stderr, flush=True)
            sys.exit(1)
        return self.multiplyFeatures(sentence, featVec)

    def multiplyFeatures(self, sentence, featVec):
        multipliedFeatVec = []
        sentenceLen = len(sentence)
        for c in range(sentenceLen):
            multipliedFeatVec.append([])
            # Iterate the radius, but keep the bounds of the list!
            for pos in range(max(c - self.radius, 0),
                             min(c + self.radius + 1, sentenceLen)):
                for feat in featVec[pos]:
                    if feat != 0:
                        multipliedFeatVec[c].append('{0}_{1}={2}'.format(
                            str(pos - c), self.name, str(feat)))
        return multipliedFeatVec


class Lexicon:
    """
    the Lexicon class generates so-called lexicon features
    an instance of Lexicon() should be initialized for each lexicon file
    """
    def __init__(self, inputFile):
        self.phraseList = set()
        self.endParts = set()
        self.midParts = set()
        self.startParts = set()
        for line in open(inputFile, encoding='UTF-8'):
            phrase = line.strip()
            self.phraseList.add(phrase)
            words = phrase.split()
            if len(words) > 1:
                self.endParts.add(words[-1])
                self.startParts.add(words[0])
                if len(words) > 2:
                    for w in words[1:-1]:
                        self.midParts.add(w)

    def getWordFeats(self, word):
        wordFeats = []
        if word in self.phraseList:
            wordFeats.append('lone')
        if word in self.endParts:
            wordFeats.append('end')
        if word in self.startParts:
            wordFeats.append('start')
        if word in self.midParts:
            wordFeats.append('mid')
        return wordFeats

    def lexEvalSentence(self, sentence):
        return [self.getWordFeats(word) for word in sentence]
