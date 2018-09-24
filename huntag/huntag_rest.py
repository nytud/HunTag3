#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import sys
import codecs
from collections import defaultdict

from flask import Flask, request, abort, Response, stream_with_context
from flask_restful import Resource, Api

# Import Tagger
from huntag.tagger import Tagger
from huntag.transmodel import TransModel
from huntag.tools import get_featureset_yaml, data_sizes

# import atexit
import threading

# ################################################# ONLY FOR DEVEL!!! ##################################################

dev_model_name = 'test'
dev_cfg_file = 'configs/maxnp.szeged.hfst.yaml'

# ################################################# ONLY FOR DEVEL!!! #################################################

# lock to control access to variable
huntag_lock = threading.Lock()
# atexit.register(huntag_tagger.__del__)

app = Flask(__name__)
api = Api(app)


class HunTagREST(Resource):
    @staticmethod
    @app.route('/')
    @app.route('/tag_file')
    def get():
        return 'Usage: HTTP POST /tag_file a file in the apropriate format'

    def post(self):
        if 'file' not in request.files:
            abort(400)
        inp_file = codecs.getreader('UTF-8')(request.files['file'])
        with huntag_lock:
            return Response(stream_with_context(self._tagger.tag_file_and_write_as_stream(inp_file)),
                            direct_passthrough=True)

    def __init__(self, tagger=None, model_name=None, cfg_file=None):
        # Init options from huntag_main.py or here...
        if tagger is None:
            # When initialized here, one must supply the model_name and the cfg_file!
            if model_name is None or cfg_file is None:
                print('No model_name ({0}) or cfg_file ({1}) given!'.format(model_name, cfg_file), file=sys.stderr)
                exit(1)

            options = {'model_filename': '{0}{1}'.format(model_name, '.model'),
                       'featcounter_filename': '{0}{1}'.format(model_name, '.featureNumbers.gz'),
                       'labelcounter_filename': '{0}{1}'.format(model_name, '.labelNumbers.gz'),
                       'tag_field': 'label',
                       'data_sizes': data_sizes,
                       'field_names': defaultdict(str),  # Dummy variable
                       'cfg_file': cfg_file,
                       'transmodel_filename': '{0}{1}'.format(model_name, '.transmodel')}

            # Load models and initiate the tagger
            print('loading transition model...', end='', file=sys.stderr, flush=True)
            trans_model = TransModel.load_from_file(options['transmodel_filename'])
            print('done', file=sys.stderr, flush=True)
            feature_set = get_featureset_yaml(cfg_file)
            self._tagger = Tagger(feature_set, trans_model, options)
        else:
            self._tagger = tagger


def add_params(tagger=None, model_name=None, cfg_file=None):
    # To bypass using self and @route together TODO: Better solution?
    api.add_resource(HunTagREST, '/tag_file', resource_class_kwargs={'tagger': tagger, 'model_name': model_name,
                                                                     'cfg_file': cfg_file})


if __name__ == '__main__':
    add_params(model_name=dev_model_name, cfg_file=dev_cfg_file)
    app.run(debug=True)
