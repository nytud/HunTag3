#!/usr/bin/env python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import sys
from os import listdir
from os.path import join as os_path_join

from . import Trainer, Tagger, TransModel, parse_args

from xtsv import process, parser_skeleton, jnius_config, build_pipeline


def main():
    argparser = parser_skeleton(description='HunTag3 - A sequential tagger for NLP combining'
                                            ' the Scikit-learn/LinearRegressionClassifier linear classifier'
                                            ' and Hidden Markov Models')
    opts = parse_args(argparser)

    jnius_config.classpath_show_warning = opts.verbose  # Suppress warning.

    # Set input and output iterators...
    if opts.input_text is not None:
        print('Sorry, --text is not available!', file=sys.stderr)
        sys.exit(1)
    else:
        input_data = opts.input_stream
    output_iterator = opts.output_stream

    options = vars(opts)

    # Set the tagger name as in the tools dictionary
    used_tools = ['huntag']
    presets = []

    # Init and run the module as it were in xtsv

    # The relevant part of config.py
    huntag_tagger = ('huntag', 'Tagger', 'HunTag3 (emNER, emChunk)', (options,),
                     {'source_fields': set(), 'target_fields': [opts.label_tag_field]})
    tools = [(huntag_tagger, ('huntag', 'HunTag3'))]

    if options['task'] == 'transmodel-train':  # TRANSMODEL TRAIN

        trans_model = TransModel(source_fields={options['gold_tag_field']}, lmw=options['lmw'],
                                 order=options['transmodel_order'])

        # It's possible to train multiple times incrementally... (Just call process on different data, then compile())
        # Exhaust training process iterator...
        for _ in process(input_data, trans_model):
            pass

        # Close training, compute probabilities
        trans_model.compile()
        trans_model.save_to_file(options['transmodel_filename'])
    elif options['task'] in {'train', 'most-informative-features', 'train-featurize'}:  # TRAIN

        trainer = Trainer(options, source_fields={options['gold_tag_field']})

        # Exhaust training process iterator...
        for _ in process(input_data, trainer):
            pass
        trainer.cutoff_feats()

        if options['task'] == 'most-informative-features':
            trainer.most_informative_features(output_iterator)
        elif options['task'] == 'train-featurize':
            trainer.write_featurized_input(output_iterator)
        else:
            trainer.train()
            trainer.save()

    elif options['task'] in {'print-weights', 'tag-featurize'}:  # TAG (minus real tagging handled by xtsv)

        tagger = Tagger(options, target_fields=[options['label_tag_field']])

        if options['io_dirs'] is not None:  # Tag all files in a directory file to to filename.tagged
            inp_dir, out_dir = options['io_dirs']
            for fn in listdir(inp_dir):
                print('processing file {0}...'.format(fn), end='', file=sys.stderr, flush=True)
                with open(os_path_join(inp_dir, fn), encoding='UTF-8') as ifh, \
                        open(os_path_join(out_dir, '{0}.tagged'.format(fn)), 'w', encoding='UTF-8') as ofh:
                    ofh.writelines(process(ifh, tagger))
        elif options['task'] == 'print-weights':  # Print MaxEnt weights to output stream
            tagger.print_weights(output_iterator, options['num_weights'])
    else:  # options['task'] == tag
        # Tag a featurized or unfeaturized file or write the featurized format to to output_stream
        # Run the pipeline on input and write result to the output...
        output_iterator.writelines(build_pipeline(input_data, used_tools, tools, presets, opts.conllu_comments))

    # TODO this method is recommended when debugging the tool
    # Alternative: Run specific tool for input (still in emtsv format):
    # from xtsv import process
    # from emdummy import EmDummy
    # output_iterator.writelines(process(input_data, EmDummy(*em_dummy[3], **em_dummy[4])))

    # Alternative2: Run REST API debug server
    # from xtsv import pipeline_rest_api, singleton_store_factory
    # app = pipeline_rest_api('TEST', tools, {},  conll_comments=False, singleton_store=singleton_store_factory(),
    #                         form_title='TEST TITLE', doc_link='https://github.com/dlt-rilmta/emdummy')
    # app.run()


if __name__ == '__main__':
    main()
