#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import argparse
from os.path import isdir, join, isfile
import sys
import os
import numpy as np
import yaml

from feature import Feature
from trainer import Trainer
from tagger import Tagger
from transmodel import TransModel


def main_trans_model_train(options):
    trans_model = TransModel(options['tag_field'], lmw=options['lmw'], order=options['transmodel_order'])
    # It's possible to train multiple times incrementally... (Just call this function on different data, then compile())
    trans_model.train(options['input_stream'])
    # Close training, compute probabilities
    trans_model.compile()
    trans_model.save_to_file(options['transmodel_filename'])


def main_train(feature_set, options):
    trainer = Trainer(feature_set, options)

    trainer.get_events_from_data(options['input_stream'], options['inp_featurized'])
    trainer.cutoff_feats()

    if options['task'] == 'most-informative-features':
        trainer.most_informative_features(options['output_stream'])
    elif options['task'] == 'train-featurize':
        trainer.write_featurized_input(options['output_stream'])
    else:
        trainer.train()
        trainer.save()


def main_tag(feature_set, options):
    if options['task'] not in {'print_weights', 'tag-featurize'}:
        print('loading transition model...', end='', file=sys.stderr, flush=True)
        trans_model = TransModel.load_from_file(options['transmodel_filename'])
        print('done', file=sys.stderr, flush=True)
    else:
        trans_model = None

    tagger = Tagger(feature_set, trans_model, options)

    if 'io_dirs' in options and options['io_dirs']:  # Tag all files in a directory file to to filename.tagged
        for sen, filename in tagger.tag_dir(options['io_dirs'][0]):
            write_sentence(sen, open(join(options['io_dirs'][1], '{0}.tagged'.format(filename)), 'a', encoding='UTF-8'))
    elif 'print_weights' in options and options['print_weights']:  # Print MaxEnt weights to STDOUT
        tagger.print_weights(options['print_weights'], options['output_stream'])
    else:  # Tag a featurized or unfeaturized file or write the featurized format to to output_stream
        for sen, comment in tagger.tag_corp(options['input_stream'], options['inp_featurized'],
                                            options['task'] == 'tag-featurize'):
            write_sentence(sen, options['output_stream'], comment)


def write_sentence(sen, out=sys.stdout, comment=None):
    if comment:
        out.write('{0}\n'.format(comment))
    out.writelines('{0}\n'.format('\t'.join(tok)) for tok in sen)
    out.write('\n')
    out.flush()


def load_yaml(cfg_file):
    lines = open(cfg_file, encoding='UTF-8').readlines()
    try:
        start = lines.index('%YAML 1.1\n')
    except ValueError:
        print('Error in config file: No document start marker found!', file=sys.stderr)
        sys.exit(1)
    rev = lines[start:]
    rev.reverse()
    try:
        end = rev.index('...\n')*(-1)
    except ValueError:
        print('Error in config file: No document end marker found!', file=sys.stderr)
        sys.exit(1)
    if end == 0:
        lines = lines[start:]
    else:
        lines = lines[start:end]

    return yaml.load(''.join(lines))


def get_featureset_yaml(cfg_file):
    features = {}
    default_radius = -1
    default_cutoff = 1
    cfg = load_yaml(cfg_file)

    if 'default' in cfg:
        default_cutoff = cfg['default'].get('cutoff', default_cutoff)
        default_radius = cfg['default'].get('radius', default_radius)

    for feat in cfg['features']:
        options = feat.get('options', {})

        if isinstance(feat['fields'], int):
            fields = [feat['fields']]
        else:
            fields = [int(field) for field in feat['fields'].split(',')]

        radius = feat.get('radius', default_radius)
        cutoff = feat.get('cutoff', default_cutoff)
        name = feat['name']
        features[name] = Feature(feat['type'], name, feat['action_name'], fields, radius, cutoff, options)

    return features


def valid_dir(input_dir):
    if not isdir(input_dir):
        raise argparse.ArgumentTypeError('"{0}" must be a directory!'.format(input_dir))
    out_dir = '{0}_out'.format(input_dir)
    os.mkdir(out_dir)
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
                             'print-weights, train-featurize, tag-featurize')

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

    parser.add_argument('-t', '--tag-field', dest='tag_field', type=int, default=-1,
                        help='specify FIELD containing the labels to build models from',
                        metavar='FIELD')

    parser.add_argument('--input-featurized', dest='inp_featurized', action='store_true', default=False,
                        help='use training events in FILE (already featurized input, see {train,tag}-featurize)')

    parser.add_argument('-o', '--output', dest='output_filename',
                        help='Use output file instead of STDOUT',
                        metavar='FILE')

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

    options.model_filename = '{0}{1}'.format(options.model_name, options.model_ext)
    options.transmodel_filename = '{0}{1}'.format(options.model_name, options.transmodel_ext)
    options.featcounter_filename = '{0}{1}'.format(options.model_name, options.featurenumbers_ext)
    options.labelcounter_filename = '{0}{1}'.format(options.model_name, options.labelnumbers_ext)

    # Data sizes across the program (training and tagging). Check manuals for other sizes
    options.data_sizes = {'rows': 'Q', 'rows_np': np.uint64,         # Really big...
                          'cols': 'Q', 'cols_np': np.uint64,         # ...enough for indices
                          'data': 'B', 'data_np': np.uint8,          # Currently data = {0, 1}
                          'labels': 'H', 'labels_np': np.uint16,     # Currently labels > 256...
                          'sent_end': 'Q', 'sent_end_np': np.uint64  # Sentence Ends in rowIndex
                          }                                          # ...for safety

    options_dict = vars(options)
    if options_dict['input_filename']:
        options_dict['input_stream'] = open(options_dict['input_filename'], encoding='UTF-8')
    else:
        options_dict['input_stream'] = sys.stdin

    if options_dict['output_filename']:
        options_dict['output_stream'] = open(options_dict['output_filename'], 'w', encoding='UTF-8')
    else:
        options_dict['output_stream'] = sys.stdout

    # Use with featurized input or raw input
    if options_dict['inp_featurized']:
        feature_set = None
    else:
        feature_set = get_featureset_yaml(options_dict['cfg_file'])

    if options_dict['task'] == 'transmodel-train':
        main_trans_model_train(options_dict)
    elif options_dict['task'] in {'train', 'most-informative-features', 'train-featurize'}:
        main_train(feature_set, options_dict)
    elif options_dict['task'] in {'tag', 'print-weights', 'tag-featurize'}:
        main_tag(feature_set, options_dict)
    else:  # Will never happen because argparse...
        print('Error: Task name must be specified! Please see --help!', file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
