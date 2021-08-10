#!/usr/bin/env python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import sys
from os import mkdir
from argparse import ArgumentTypeError, ArgumentParser
from os.path import isdir, isfile, abspath, dirname, join

import yaml
import numpy as np

from huntag.feature import Feature


def valid_dir(input_dir):
    if not isdir(input_dir):
        raise ArgumentTypeError('"{0}" must be a directory!'.format(input_dir))
    out_dir = '{0}_out'.format(input_dir)
    mkdir(out_dir)
    return input_dir, out_dir


def valid_file(input_file_name):  # Config and model file is also searched relatve to the module directory
    for input_file in (input_file_name, join(dirname(abspath(__file__)), input_file_name)):
        if isfile(input_file):
            return input_file
    else:
        raise ArgumentTypeError('"{0}" must be a file!'.format(input_file_name))


def parse_args(parser=ArgumentParser()):
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

    parser.add_argument('-g', '--gold-tag-field', dest='gold_tag_field', default='gold',
                        help='specify FIELD containing the gold labels to build models from (training)',
                        metavar='FIELD')

    parser.add_argument('-l', '--label-tag-field', dest='label_tag_field', default='label',
                        help='specify FIELD where the predicted labels are requested (tagging)',
                        metavar='FIELD')

    parser.add_argument('--input-featurized', dest='inp_featurized', action='store_true', default=False,
                        help='use training events in FILE (already featurized input, see {train,tag}-featurize)')

    parser.add_argument('-w', '--num-weights', dest='num_weights', type=int, default=100,
                        help='Print only the first N weights',
                        metavar='N')

    parser.add_argument('-d', '--input-dir', dest='io_dirs', type=valid_dir,
                        help='process all files in DIR (instead of stdin)',
                        metavar='DIR')

    options = parser.parse_args()

    if options.input_stream != sys.stdin and options.io_dirs is not None:
        print('Error: -i/--input and -d/--input-dir are mutually exclusive arguments!', file=sys.stderr)
        sys.exit(1)

    # Put together model filenames...
    options.model_filename = '{0}{1}'.format(options.model_name, options.model_ext)
    options.featcounter_filename = '{0}{1}'.format(options.model_name, options.featurenumbers_ext)
    options.labelcounter_filename = '{0}{1}'.format(options.model_name, options.labelnumbers_ext)
    options.transmodel_filename = '{0}{1}'.format(options.model_name, options.transmodel_ext)

    if options.inp_featurized and options.task in {'train-featurize', 'tag-featurize'}:
        print('Error: Can not featurize input, which is already featurized according to CLI options!', file=sys.stderr,
              flush=True)
        sys.exit(1)

    return options


data_sizes = {'rows': 'Q', 'rows_np': np.uint64,         # Really big...
              'cols': 'Q', 'cols_np': np.uint64,         # ...enough for indices
              'data': 'B', 'data_np': np.uint8,          # Currently data = {0, 1}
              'labels': 'H', 'labels_np': np.uint16,     # Currently labels > 256...
              'sent_end': 'Q', 'sent_end_np': np.uint64  # Sentence Ends in rowIndex
              }                                          # ...for safety


def load_options_and_features(opts, source_fields, target_fields):
    # Load default options
    options = {'data_sizes': data_sizes, 'task': 'tag', 'inp_featurized': False,
               'model_filename': '{0}{1}'.format(opts['model_name'], '.model'),
               'featcounter_filename': '{0}{1}'.format(opts['model_name'], '.featureNumbers.gz'),
               'labelcounter_filename': '{0}{1}'.format(opts['model_name'], '.labelNumbers.gz'),
               'transmodel_filename': '{0}{1}'.format(opts['model_name'], '.transmodel')}

    options.update(opts)  # Update defaults with supplied options

    # Load features
    if options['inp_featurized']:  # Use with featurized input or raw input
        features = None
    elif 'features' not in options:  # Load features
        features = get_featureset_yaml(options['cfg_file'])
    else:
        features = options['features']  # Or feed loaded features!

    # Field names for e-magyar TSV
    if source_fields is None:
        source_fields = set()
    if target_fields is None:
        target_fields = []

    source_fields = source_fields
    target_fields = target_fields

    source_fields = source_fields.union({field for feat in features.values() for field in feat.fields})
    return features, source_fields, target_fields, options


def load_yaml(cfg_file):
    try:
        with open(cfg_file, encoding='UTF-8') as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        print('Error: Config file ({0}) not found!'.format(cfg_file), file=sys.stderr)
        sys.exit(1)

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

    return yaml.load(''.join(lines), Loader=yaml.SafeLoader)


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

        fields = feat['fields'].split(',')

        radius = feat.get('radius', default_radius)
        cutoff = feat.get('cutoff', default_cutoff)
        name = feat['name']

        if feat['type'] == 'lex':  # Fix path for lexicon files if needed
            feat['action_name'] = valid_file(feat['action_name'])

        features[name] = Feature(feat['type'], name, feat['action_name'], fields, radius, cutoff, options)

    return features
