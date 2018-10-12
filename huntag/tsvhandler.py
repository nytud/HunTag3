#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import sys


def process_header(stream, target_fields):
    fields = stream.readline().strip().split()                      # Read header to fields
    fields.extend(target_fields)                                    # Add target fields
    field_names = {name: i for i, name in enumerate(fields)}        # Decode field names
    field_names.update({i: name for i, name in enumerate(fields)})  # Both ways...
    return '{0}\n'.format('\t'.join(fields)), field_names


def process(stream, internal_app):
    header, field_names = process_header(stream, internal_app.target_fields)
    yield header

    field_values = internal_app.prepare_fields(field_names)

    print('featurizing sentences...', end='', file=sys.stderr, flush=True)
    sen_count = 0
    for sen_count, (sen, comment) in enumerate(sentence_iterator(stream)):
        sen_count += 1
        if comment:
            yield '{0}\n'.format(comment)

        yield from ('{0}\n'.format('\t'.join(tok)) for tok in internal_app.process_sentence(sen, field_values))
        yield '\n'

        if sen_count % 1000 == 0:
            print('{0}...'.format(sen_count), end='', file=sys.stderr, flush=True)
    print('{0}...done'.format(sen_count), file=sys.stderr, flush=True)


def sentence_iterator(input_stream):
    curr_sen = []
    curr_comment = None
    for line in input_stream:
        line = line.strip()
        # Comment handling
        if line.startswith('"""'):
            if len(curr_sen) == 0:  # Comment before sentence
                curr_comment = line
            else:  # Error: Comment in the middle of sentence
                print('ERROR: comments are only allowed before a sentence!', file=sys.stderr, flush=True)
                sys.exit(1)
        # Blank line handling
        elif len(line) == 0:
            if curr_sen:  # End of sentence
                yield curr_sen, curr_comment
                curr_sen = []
                curr_comment = None
            else:  # WARNING: Multiple blank line
                print('WARNING: wrong formatted sentences, only one blank line allowed!', file=sys.stderr, flush=True)
        else:
            curr_sen.append(line.split())
    if curr_sen:
        print('WARNING: No blank line before EOF!', file=sys.stderr, flush=True)
        yield curr_sen, curr_comment
