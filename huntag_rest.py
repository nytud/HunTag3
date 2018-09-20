#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import sys
from collections import defaultdict

from flask import Flask, request, abort, Response
from flask_restful import Resource, Api

# Import Tagger
from huntag.tagger import Tagger
from huntag.transmodel import TransModel
from huntag.tools import get_featureset_yaml, feature_names_to_indices, data_sizes

# import atexit
import threading

# TO BE MODIFIED!!!
model_name = 'hfst.maxnp'
cfg_file = 'configs/maxnp.szeged.hfst.yaml'

# THESE SHOULD WORK WITHOUT MODIFICATION!
options = {'model_filename': '{0}{1}'.format(model_name, '.model'),
           'featcounter_filename': '{0}{1}'.format(model_name, '.featureNumbers.gz'),
           'labelcounter_filename': '{0}{1}'.format(model_name, '.labelNumbers.gz'),
           'tag_field': 'label',
           'data_sizes': data_sizes,
           'field_names': defaultdict(str)}  # Dummy variable
transmodel_filename = '{0}{1}'.format(model_name, '.transmodel')

# lock to control access to variable
huntag_lock = threading.Lock()

# Initiate
print('loading transition model...', end='', file=sys.stderr, flush=True)
trans_model = TransModel.load_from_file(transmodel_filename)
print('done', file=sys.stderr, flush=True)
feature_set = get_featureset_yaml(cfg_file)
huntag_tagger = Tagger(feature_set, trans_model, options)

# atexit.register(huntag_tagger.__del__)

app = Flask(__name__)
api = Api(app)


def tag_file_and_write_as_stream(inp_file):
    first_line = inp_file.readline()
    yield first_line
    field_names = {name: i for i, name in enumerate(first_line.strip().split())}
    huntag_tagger._features = feature_names_to_indices(feature_set, field_names)
    huntag_tagger._tag_field = field_names[options['tag_field']]

    for sen, comment in huntag_tagger.tag_corp(inp_file):
        if comment:
            yield '{0}\n'.format(comment)
        yield from ('{0}\n'.format('\t'.join(tok)) for tok in sen)
        yield '\n'


class HunTagREST(Resource):
    @staticmethod
    @app.route('/')
    def usage():
            return 'Usage: HTTP POST on /tag'

    @staticmethod
    @app.route('/tag_file')
    def tag_file_usage():
            return 'Usage: HTTP POST /tag a file in the apropriate format'

    @staticmethod
    @app.route('/tag_file', methods=['POST'])
    def tag_file():
        if 'file' not in request.files:
            abort(400)
        inp_file = request.files['file']

        return Response(tag_file_and_write_as_stream(inp_file), direct_passthrough=True)


if __name__ == '__main__':
    app.run(debug=False)
