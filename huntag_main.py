#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import os
import sys
import argparse
from os import mkdir
from os.path import isdir, join, isfile
from collections import defaultdict

from huntag.tools import get_featureset_yaml, data_sizes
from huntag.tsvhandler import process
from huntag.trainer import Trainer
from huntag.tagger import Tagger
from huntag.transmodel import TransModel


def main_trans_model_train(input_stream, model_filename, gold_tag_field, lm_weight, model_order):
    trans_model = TransModel(gold_tag_field, lmw=lm_weight, order=model_order)
    # It's possible to train multiple times incrementally... (Just call this function on different data, then compile())

    # Exhaust training process iterator...
    for _ in process(input_stream, trans_model):
        pass

    # Close training, compute probabilities
    trans_model.compile()
    trans_model.save_to_file(model_filename)


def main_train(task, input_stream, output_stream, feature_set, options):
    trainer = Trainer(feature_set, options)

    # Exhaust training process iterator...
    for _ in process(input_stream, trainer):
        pass
    trainer.cutoff_feats()

    if task == 'most-informative-features':
        trainer.most_informative_features(output_stream)
    elif task == 'train-featurize':
        trainer.write_featurized_input(output_stream)
    else:
        trainer.train()
        trainer.save()


def main_tag(task, input_stream, output_stream, feature_set, options, print_weights=None, io_dirs=None):
    tagger = Tagger(feature_set, options)

    if io_dirs is not None:  # Tag all files in a directory file to to filename.tagged
        tag_dir(io_dirs, tagger)
    elif task == 'print-weights':  # Print MaxEnt weights to output stream
        tagger.print_weights(output_stream, print_weights)
    else:  # Tag a featurized or unfeaturized file or write the featurized format to to output_stream
        output_stream.writelines(process(input_stream, tagger))
        output_stream.flush()


def tag_dir(io_dirs, tagger):
    inp_dir, out_dir = io_dirs
    for fn in os.listdir(inp_dir):
        print('processing file {0}...'.format(fn), end='', file=sys.stderr, flush=True)
        with open(os.path.join(inp_dir, fn), encoding='UTF-8') as ifh,\
                open(join(out_dir, '{0}.tagged'.format(fn)), 'w', encoding='UTF-8') as ofh:
                    ofh.writelines(process(ifh, tagger))


def valid_dir(input_dir):
    if not isdir(input_dir):
        raise argparse.ArgumentTypeError('"{0}" must be a directory!'.format(input_dir))
    out_dir = '{0}_out'.format(input_dir)
    mkdir(out_dir)
    return input_dir, out_dir


def valid_file(input_file):
    if not isfile(input_file):
        raise argparse.ArgumentTypeError('"{0}" must be a file!'.format(input_file))
    return input_file


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('task', choices=['transmodel-train', 'most-informative-features', 'train', 'tag',
                                         'print-weights', 'train-featurize', 'tag-featurize'],
                        help='avaliable tasks: transmodel-train, most-informative-features, train, tag, '
                             'print-weights, train-featurize, tag-featurize)')

    parser.add_argument('-c', '--config-file', dest='cfg_file', type=valid_file,
                        help='read feature configuration from FILE',
                        metavar='FILE')

    parser.add_argument('-m', '--model', dest='model_name', required=True,
                        help='name of the (trans) model to be read/written',
                        metavar='NAME')

    parser.add_argument('--model-ext', dest='model_ext', default='.model',
                        help='extension of model to be read/written',
                        metavar='EXT')

    parser.add_argument('--trans-model-ext', dest='transmodel_ext', default='.transmodel',
                        help='extension of trans model file to be read/written',
                        metavar='EXT')

    parser.add_argument('--trans-model-order', dest='transmodel_order', default=3,
                        help='order of the transition model',
                        metavar='EXT')

    parser.add_argument('--feat-num-ext', dest='featurenumbers_ext', default='.featureNumbers.gz',
                        help='extension of feature numbers file to be read/written',
                        metavar='EXT')

    parser.add_argument('--label-num-ext', dest='labelnumbers_ext', default='.labelNumbers.gz',
                        help='extension of label numbers file to be read/written',
                        metavar='EXT')

    parser.add_argument('-l', '--language-model-weight', dest='lmw',
                        type=float, default=1,
                        help='set relative weight of the language model to L',
                        metavar='L')

    parser.add_argument('-O', '--cutoff', dest='cutoff', type=int, default=1,
                        help='set global cutoff to C',
                        metavar='C')

    parser.add_argument('-p', '--parameters', dest='train_params',
                        help='pass PARAMS to trainer',
                        metavar='PARAMS')

    parser.add_argument('-u', '--used-feats', dest='used_feats', type=valid_file,
                        help='limit used features to those in FILE',
                        metavar='FILE')

    parser.add_argument('-t', '--tag-field', dest='tag_field', default='label',
                        help='specify FIELD where the generated labels requested (tagging)',
                        metavar='FIELD')

    parser.add_argument('-g', '--gold-tag-field', dest='gold_tag_field', default='gold',
                        help='specify FIELD containing the gold labels to build models from (training)',
                        metavar='FIELD')

    parser.add_argument('--input-featurized', dest='inp_featurized', action='store_true', default=False,
                        help='use training events in FILE (already featurized input, see {train,tag}-featurize)')

    parser.add_argument('-o', '--output', dest='output_filename',
                        help='Use output file instead of STDOUT',
                        metavar='FILE')

    parser.add_argument('-w', '--num-weights', dest='num_weights', type=int, default=100,
                        help='Print only the first N weights',
                        metavar='N')

    group_i = parser.add_mutually_exclusive_group()

    group_i.add_argument('-i', '--input', dest='input_filename', type=valid_file,
                         help='Use input file instead of STDIN',
                         metavar='FILE')

    group_i.add_argument('-d', '--input-dir', dest='io_dirs', type=valid_dir,
                         help='process all files in DIR (instead of stdin)',
                         metavar='DIR')

    return parser.parse_args()


def main():
    options = parse_args()
    if options.inp_featurized and options.task in {'train-featurize', 'tag-featurize'}:
        print('Error: Can not featurize input, which is already featurized according to CLI options!', file=sys.stderr,
              flush=True)
        sys.exit(1)

    options.data_sizes = data_sizes

    # Model files...
    options.model_filename = '{0}{1}'.format(options.model_name, options.model_ext)
    options.featcounter_filename = '{0}{1}'.format(options.model_name, options.featurenumbers_ext)
    options.labelcounter_filename = '{0}{1}'.format(options.model_name, options.labelnumbers_ext)
    transmodel_filename = '{0}{1}'.format(options.model_name, options.transmodel_ext)
    options.transmodel_filename = transmodel_filename

    task = options.task

    # Set input and output stream...
    input_filename = options.input_filename
    output_filename = options.output_filename
    if input_filename:
        input_stream = open(input_filename, encoding='UTF-8')
    else:
        input_stream = sys.stdin

    if output_filename:
        output_stream = open(output_filename, 'w', encoding='UTF-8')
    else:
        output_stream = sys.stdout

    options_dict = vars(options)

    # Use with featurized input or raw input
    if options_dict['inp_featurized'] or task == 'transmodel-train':
        feature_set = None
        options_dict['field_names'] = defaultdict(str)
    else:
        feature_set = get_featureset_yaml(options_dict['cfg_file'])

    if task == 'transmodel-train':
        main_trans_model_train(input_stream, transmodel_filename, options_dict['gold_tag_field'], options_dict['lmw'],
                               options_dict['transmodel_order'])
    elif task in {'train', 'most-informative-features', 'train-featurize'}:
        main_train(task, input_stream, output_stream, feature_set, options_dict)
    elif task in {'tag', 'print-weights', 'tag-featurize'}:
        main_tag(task, input_stream, output_stream, feature_set, options_dict, options_dict['num_weights'],
                 options_dict['io_dirs'])
    else:  # Will never happen because argparse...
        print('Error: Task name must be specified! Please see --help!', file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
