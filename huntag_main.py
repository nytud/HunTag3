#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import os
import sys
import argparse
from os import mkdir
from os.path import isdir, join, isfile

from xtsv.tsvhandler import process
from huntag.trainer import Trainer
from huntag.tagger import Tagger
from huntag.transmodel import TransModel


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

    parser.add_argument('--language-model-weight', dest='lmw',  type=float, default=1,
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

    parser.add_argument('-l', '--label-tag-field', dest='label_tag_field', default='label',
                        help='specify FIELD containing the predicted labels (tagging)',
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

    options = parser.parse_args()

    # Put together model filenames...
    options.model_filename = '{0}{1}'.format(options.model_name, options.model_ext)
    options.featcounter_filename = '{0}{1}'.format(options.model_name, options.featurenumbers_ext)
    options.labelcounter_filename = '{0}{1}'.format(options.model_name, options.labelnumbers_ext)
    options.transmodel_filename = '{0}{1}'.format(options.model_name, options.transmodel_ext)

    # Set input and output stream...
    if options.input_filename:
        options.input_stream = open(options.input_filename, encoding='UTF-8')
    else:
        options.input_stream = sys.stdin

    if options.output_filename:
        options.output_stream = open(options.output_filename, 'w', encoding='UTF-8')
    else:
        options.output_stream = sys.stdout

    if options.inp_featurized and options.task in {'train-featurize', 'tag-featurize'}:
        print('Error: Can not featurize input, which is already featurized according to CLI options!', file=sys.stderr,
              flush=True)
        sys.exit(1)

    return vars(options)


def main():
    options = parse_args()

    if options['task'] == 'transmodel-train':  # TRANSMODEL TRAIN

        trans_model = TransModel(source_fields={options['gold_tag_field']}, lmw=options['lmw'],
                                 order=options['transmodel_order'])

        # It's possible to train multiple times incrementally... (Just call process on different data, then compile())
        # Exhaust training process iterator...
        for _ in process(options['input_stream'], trans_model):
            pass

        # Close training, compute probabilities
        trans_model.compile()
        trans_model.save_to_file(options['transmodel_filename'])
    elif options['task'] in {'train', 'most-informative-features', 'train-featurize'}:  # TRAIN

        trainer = Trainer(options, source_fields={options['gold_tag_field']})

        # Exhaust training process iterator...
        for _ in process(options['input_stream'], trainer):
            pass
        trainer.cutoff_feats()

        if options['task'] == 'most-informative-features':
            trainer.most_informative_features(options['output_stream'])
        elif options['task'] == 'train-featurize':
            trainer.write_featurized_input(options['output_stream'])
        else:
            trainer.train()
            trainer.save()

    elif options['task'] in {'tag', 'print-weights', 'tag-featurize'}:  # TAG

        tagger = Tagger(options, target_fields=[options['label_tag_field']])

        if options['io_dirs'] is not None:  # Tag all files in a directory file to to filename.tagged
            tag_dir(options['io_dirs'], tagger)
        elif options['task'] == 'print-weights':  # Print MaxEnt weights to output stream
            tagger.print_weights(options['output_stream'], options['num_weights'])
        else:  # Tag a featurized or unfeaturized file or write the featurized format to to output_stream
            options['output_stream'].writelines(process(options['input_stream'], tagger))
            options['output_stream'].flush()

    else:  # Will never happen because argparse...
        print('Error: Task name must be specified! Please see --help!', file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
