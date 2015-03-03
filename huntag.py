#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

from collections import defaultdict
import argparse
from os.path import isdir, join
import sys
import os
import numpy as np

from feature import Feature
from trainer import Trainer
from tagger import Tagger
from bigram import Bigram
from tools import BookKeeper, writeSentence, sentenceIterator


def main_bigramTrain(options, inputStream):
    bigramModel = Bigram(0.000000000000001)
    for sen, _ in sentenceIterator(inputStream):
        tags = [tok[options.tagField] for tok in sen]
        bigramModel.obsSequence(tags)
    bigramModel.count()
    bigramModel.writeToFile(options.bigramModelFileName)


def main_train(featureSet, options, inputStream=sys.stdin):
    optionsDict = vars(options)
    if options.usedFeats:
        optionsDict['usedFeats'] = open(options.usedFeats, encoding='UTF-8')
    trainer = Trainer(featureSet, optionsDict)
    if options.inFeatFileName:
        trainer.getEventsFromFile(options.inFeatFileName)
    else:
        trainer.getEvents(inputStream, options.outFeatFileName)
    trainer.cutoffFeats()
    trainer.train()
    trainer.save()


def main_tag(featureSet, options):
    optionsDict = vars(options)
    optionsDict['labelCounter'], optionsDict['featCounter'] = BookKeeper(), BookKeeper()
    optionsDict['labelCounter'].readFromFile(optionsDict['labelCounterFileName'])
    optionsDict['featCounter'].readFromFile(optionsDict['featCounterFileName'])
    tagger = Tagger(featureSet, optionsDict)
    if options.inFeatFileName:
        tagger_func = lambda: tagger.tag_features(options.inFeatFileName)
        writer_func = lambda s, c: writeSentence(s, comment=c)
    elif options.io_dirs:
        tagger_func = lambda: tagger.tag_dir(options.io_dirs[0])
        writer_func = lambda s, c: writeSentence(s, out=open(join(options.io_dirs[1],
            '{0}.tagged'.format(c)), 'a', encoding='UTF-8'))
    else:
        tagger_func = lambda: tagger.tag_corp(sys.stdin)
        writer_func = lambda s, c: writeSentence(s, comment=c)

    for sen, other in tagger_func():
        writer_func(sen, other)


def getFeatureSet(cfgFile):
    features = {}
    optsByFeature = defaultdict(dict)
    defaultRadius = -1
    defaultCutoff = 1
    for line in open(cfgFile, encoding='UTF-8'):
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            continue
        feature = line.split()
        if feature[0] == 'let':
            featName, key, value = feature[1:4]
            optsByFeature[featName][key] = value
            continue
        if feature[0] == '!defaultRadius':
            defaultRadius = int(feature[1])
            continue
        if feature[0] == '!defaultCutoff':
            defaultCutoff = int(feature[1])
            continue

        feaType, name, actionName = feature[:3]
        fields = [int(field) for field in feature[3].split(',')]
        if len(feature) > 4:
            radius = int(feature[4])
        else:
            radius = defaultRadius
        cutoff = defaultCutoff
        options = optsByFeature[name]
        feat = Feature(feaType, name, actionName, fields, radius, cutoff, options)
        features[name] = feat

    return features


def validDir(input_dir):
    if not isdir(input_dir):
        raise argparse.ArgumentTypeError('"{0}" must be a directory!'.format(input_dir))
    out_dir = '{0}_out'.format(input_dir)
    os.mkdir(out_dir)
    return input_dir, out_dir


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('task', choices=['bigram-train', 'train', 'tag'],
                        help='avaliable tasks: bigram-train, train, tag')

    parser.add_argument('-c', '--config-file', dest='cfgFile',
                        help='read feature configuration from FILE',
                        metavar='FILE')

    parser.add_argument('-m', '--model', dest='modelName',
                        help='name of (bigram) model to be read/written',
                        metavar='NAME')

    parser.add_argument('--model-ext', dest='modelExt', default='.model',
                        help='extension of model to be read/written',
                        metavar='EXT')

    parser.add_argument('--bigram-model-ext', dest='bigramModelExt', default='.bigram',
                        help='extension of bigram model file to be read/written',
                        metavar='EXT')

    parser.add_argument('--feat-num-ext', dest='featureNumbersExt', default='.featureNumbers',
                        help='extension of feature numbers file to be read/written',
                        metavar='EXT')

    parser.add_argument('--label-num-ext', dest='labelNumbersExt', default='.labelNumbers',
                        help='extension of label numbers file to be read/written',
                        metavar='EXT')

    parser.add_argument('-l', '--language-model-weight', dest='lmw',
                        type=float, default=1,
                        help='set relative weight of the language model to L',
                        metavar='L')

    parser.add_argument('-o', '--cutoff', dest='cutoff', type=int, default=2,
                        help='set global cutoff to C',
                        metavar='C')

    parser.add_argument('-p', '--parameters', dest='trainParams',
                        help='pass PARAMS to trainer',
                        metavar='PARAMS')

    parser.add_argument('-u', '--used-feats', dest='usedFeats',
                        help='limit used features to those in FILE',
                        metavar='FILE')

    group = parser.add_mutually_exclusive_group()

    group.add_argument('-d', '--input-dir', dest='io_dirs', type=validDir,
                       help='process all files in DIR (instead of stdin)',
                       metavar='DIR')

    group.add_argument('-i', '--input-feature-file', dest='inFeatFileName',
                       help='use training events in FILE',
                       metavar='FILE')

    parser.add_argument('-f', '--feature-file', dest='outFeatFileName',
                        help='write training events to FILE',
                        metavar='FILE')

    parser.add_argument('-t', '--tag-field', dest='tagField', type=int, default=-1,
                        help='specify FIELD containing the labels to build models from',
                        metavar='FIELD')

    return parser.parse_args()


def main():
    options = parse_args()
    if not options.modelName:
        print('Error: Model name must be specified! Please see --help!', file=sys.stderr, flush=True)
        sys.exit(1)
    options.modelFileName = '{0}{1}'.format(options.modelName, options.modelExt)
    options.bigramModelFileName = '{0}{1}'.format(options.modelName, options.bigramModelExt)
    options.featCounterFileName = '{0}{1}'.format(options.modelName, options.featureNumbersExt)
    options.labelCounterFileName = '{0}{1}'.format(options.modelName, options.labelNumbersExt)

    # Data sizes across the program (training and tagging). Check manuals for other sizes
    options.dataSizes = {'rows': 'Q', 'rowsNP': np.uint64,     # Really big...
                         'cols': 'Q', 'colsNP': np.uint64,     # ...enough for indices
                         'data': 'B', 'dataNP': np.uint8,      # Currently data = {0, 1}
                         'labels': 'B', 'labelsNP': np.uint16  # Currently labels > 256...
                        }                                      # ...for safety

    if options.task == 'bigram-train':
        main_bigramTrain(options, sys.stdin)
    elif options.task == 'train':
        featureSet = getFeatureSet(options.cfgFile)
        main_train(featureSet, options)
    elif options.task == 'tag':
        if options.inFeatFileName:
            featureSet = None
        else:
            featureSet = getFeatureSet(options.cfgFile)
        main_tag(featureSet, options)
    else:
        print('Error: Task name must be specified! Please see --help!', file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
