#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import sys


def process_header(stream, source_fields, target_fields):
    fields = next(stream).strip().split('\t')                           # Read header to fields
    if not source_fields.issubset(set(fields)):
        raise NameError('Input does not have the required field names ({0}). The following field names found: {1}'.
                        format(sorted(source_fields), fields))
    fields.extend(target_fields)                                    # Add target fields when apply (only for tagging)
    field_names = {name: i for i, name in enumerate(fields)}        # Decode field names
    field_names.update({i: name for i, name in enumerate(fields)})  # Both ways...
    header = '{0}\n'.format('\t'.join(fields))
    return header, field_names


# Only This method is public...
def process(stream, internal_app, conll_comments=False):
    if len(internal_app.source_fields) > 0:
        header, field_names = process_header(stream, internal_app.source_fields, internal_app.target_fields)
        yield header

        # Like binding names to indices...
        field_values = internal_app.prepare_fields(field_names)

        print('featurizing sentences...', end='', file=sys.stderr, flush=True)
        sen_count = 0
        for sen_count, (sen, comment) in enumerate(sentence_iterator(stream, conll_comments)):
            sen_count += 1
            if len(comment) > 0:
                yield comment

            yield from ('{0}\n'.format('\t'.join(tok)) for tok in internal_app.process_sentence(sen, field_values))
            yield '\n'

            if sen_count % 1000 == 0:
                print('{0}...'.format(sen_count), end='', file=sys.stderr, flush=True)
        print('{0}...done'.format(sen_count), file=sys.stderr, flush=True)
    else:
        # This is intended to be used by the first module in the pipeline which deals with raw text (eg. tokenizer) only
        yield '{0}\n'.format('\t'.join(internal_app.target_fields))
        yield from internal_app.process_sentence(stream)


def sentence_iterator(input_stream, conll_comments=False):
    curr_sen = []
    curr_comment = ''
    for line in input_stream:
        line = line.strip()
        # Comment handling: Before sentence, line starts with # and comments are allowed by parameter
        if len(curr_sen) == 0 and line.startswith('#') and conll_comments:
                curr_comment += '{0}\n'.format(line)  # Comment before sentence
        # Blank line handling
        elif len(line) == 0:
            if curr_sen:  # End of sentence
                yield curr_sen, curr_comment
                curr_sen = []
                curr_comment = ''
            else:  # WARNING: Multiple blank line
                print('WARNING: wrong formatted sentences, only one blank line allowed!', file=sys.stderr, flush=True)
        else:
            curr_sen.append(line.split('\t'))
    if curr_sen:
        print('WARNING: No blank line before EOF!', file=sys.stderr, flush=True)
        yield curr_sen, curr_comment
