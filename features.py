#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-
"""
features.py stores implementations of individual feature types for use with
HunTag. Feel free to add your own features but please add comments to describe
them.
"""

import re
import sys

# GLOBAL DECLARATION BEGIN
# see: token_long_pattern, token_short_pattern, sentence_lemma_lowered, poss_connect, token_mmo_simple
smallcase = 'aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz'
bigcase = 'AÁBCDEÉFGHIÍJKLMNOÓÖŐPQRSTUÚÜŰVWXYZ'
big2small = {}
for i, _ in enumerate(bigcase):
    big2small[bigcase[i]] = smallcase[i]
cas_re_kr = re.compile('<CAS')
cas_re_msd = re.compile('\[?N')
possessor_msd = re.compile(r'--[sp]\d')
obj_msd = '\[?N'

possessor_kr = re.compile('<POSS')
obj_kr = 'NOUN'
mmo_patt = re.compile('[\[,\]]')
# GLOBAL DECLARATION END


# HELPER FUNCTIONS BEGIN
def tags_since_pos(sen, tok_range, my_pos, strict=True):
    """Gather all tags since POS my_pos in the sentence (not used directly as a feature)

    Args:
       sen (list): List of tokens in the sentence
       tok_range (int): range of tokens from the start of the sentence
       my_pos (str): POS tag to search for
       strict(bool): full matching or not...

    Returns:
       [tags joined by '+']: all tags since my_pos POS tag

    HunTag:
        Type: Sentence
        Field: analysis
        Example: ???
        Use case: NER, Chunk

    Replaces:
        since_dt: abstraction
    """
    tags = []
    for pos in sen[:tok_range]:
        if (strict and my_pos == pos) or (not strict and re.search(my_pos, pos)):
            tags = [pos]
        else:
            tags.append(pos)
    return ['+'.join(tags)]


def since_pos(kr_vec, c, feat_vec_elem, tag, feat_prefix):
    """Parameter XXX

    Args:
       kr_vec (list): List of tokens in the sentence
       c (int): Range of tokens from the start of the sentence
       feat_vec_elem (list): Current feature Vector element (to be updated)
       tag(str): full matching or not...
       feat_prefix(str): prefix to set

    Returns (updates featVecELem):
       [tags joined by '+']: all tags since myPos POS tag
    """
    tagst = tags_since_pos(kr_vec, c, tag)[0]
    if len(tagst) > 0:
        feat_vec_elem.append(feat_prefix + tagst)


def do_nothing(*_):
    pass


def cas_diff(kr_vec, c, feat_vec_elem, cas_re, feat_name):
    """Test if subsequent nouns is in di'cas_diff'fferent grammatical case...

    Args:
       kr_vec (list): List of tokens in the sentence
       c (int): Range of tokens from the start of the sentence
       feat_vec_elem (list): Current feature Vector element (to be updated)
       cas_re(re.pattern): pattern to match
       feat_name(str): Name of the feature to be set

    Returns (updates featVecELem):
       ['cas_diff']: if case is different...
    """
    last_f = '' if c == 0 else kr_vec[c - 1]
    if cas_re.search(last_f) and cas_re.search(kr_vec[c]) and last_f != kr_vec[c]:
        feat_vec_elem.append(feat_name)


def poss_connect(kr_vec, c, feat_vec_elem, possessor, obj, feat_prefix):
    """Connect possessor with posessed object

    Args:
       kr_vec (list): List of tokens in the sentence
       c (int): Range of tokens from the start of the sentence
       feat_vec_elem (list): Current feature Vector element (to be updated)
       possessor(re.pattern): Pattern of possessor
       obj(str): Possessed object string to be compiled into pattern
       feat_prefix(str): prefix to set

    Returns (updates featVecELem):
       [tags joined by '+']: all tags since obj POS tag
    """
    if possessor.search(kr_vec[c]):
        tagst = tags_since_pos(kr_vec, c, obj, False)[0]
        if len(tagst) > 0:
            feat_vec_elem.append(feat_prefix + tagst)
# HELPER FUNCTIONS END


# XXX Return is not bool
def token_stupid_stem(token, _=None):
    """Stem tokens with hyphen (-) in them.

    Args:
       token (str): The token
       _: Unused

    Returns:
       [Str]: The part of the token before hyphen (-) character

    HunTag:
        Type: Token
        Field: Token
        Example: 'MTI-nek' -> MTI
        Use case: NER

    Used extensively in other functions
    """
    r = token.rfind('-')
    if r == -1:
        return [token]
    else:
        return [token[:r]]


def token_has_cap_operator(form, _=None):
    """Has it capital letter anywhere?

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if lowercasing modifies the token

    HunTag:
        Type: Token
        Field: Token
        Example: 'aLma' -> [1], 'alma' -> [0]
        Use case: NER

    Replaces:
        token_isCapitalizedOperator: deprecated
        token_lowerCaseOperator: inverse, redundant
    """
    return [int(form.lower() != form)]


def token_is_cap_operator(form, _=None):
    """Only the first letter is capital
    This is the new isCapitalized: Starts with uppercase

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if lowercasing the first letter modifies the token

    HunTag:
        Type: Token
        Field: Token
        Example: 'Alma' -> [1], 'ALma' -> [0]
        Use case: NER

    Replaces:
        token_notCapitalizedOperator: inverse, redundant
    """
    return [int(form[0] != form[0].lower())]


def token_is_allcaps_operator(form, _=None):
    """StupidStem consists of uppercase letters

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if token_stupid_stem is uppercase

    HunTag:
        Type: Token
        Field: Token
        Example: 'MTI-vel' -> [1], 'Mti-vel' -> [0]
        Use case: NER
    """
    return [int(token_stupid_stem(form)[0].isupper())]


def token_is_camel_operator(form, _=None):
    """The first letter is lower, the others has, but not all uppercase (camelCasing)

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Not all letter are uppercase, but at least one is from the second character

    HunTag:
        Type: Token
        Field: Token
        Example: 'aLMa' -> [1], 'ALMA' -> [0], 'alma' -> [0]
        Use case: NER
    """
    return [int(not token_is_allcaps_operator(form) and form[1:].lower() != form[1:])]


def token_three_caps(form, _=None):
    """Token is three uppercase letters?

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Token consist of three uppercase letters

    HunTag:
        Type: Token
        Field: Token
        Example: 'MTI' -> [1], 'Mti' -> [0], 'Matáv' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int(len(form) == 3 and token_stupid_stem(form)[0].isupper())]


def token_starts_with_number_operator(form, _=None):
    """Token starts with number

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Token's first letter is a digit

    HunTag:
        Type: Token
        Field: Token
        Example: '3-gram' -> [1], 'n-gram' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int(form[0].isdigit())]


def token_has_number_operator(form, _=None):
    """Token contains numbers

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if token contains numbers

    HunTag:
        Type: Token
        Field: Token
        Example: '3-gram' -> [1], 'n-gram' -> [0]
        Use case: NER

    Replaces:
        token_isNumberOperator: deprecated
    """
    return [int(not set('0123456789').isdisjoint(set(form)))]


def token_has_dash_operator(form, _=None):
    """Token contains hyphen (-)

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if token contains hyphen (-)

    HunTag:
        Type: Token
        Field: Token
        Example: '3-gram' -> [1], 'ngram' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int('-' in form)]


def token_has_underscore_operator(form, _=None):
    """Token contains underscore (_)

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if token contains underscore (_)

    HunTag:
        Type: Token
        Field: Token
        Example: 'function_name' -> [1], 'functionName' -> [0]
        Use case: Chunk
    """
    return [int('_' in form)]


def token_has_period_operator(form, _=None):
    """Token contains period (.)

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if token contains period (.)

    HunTag:
        Type: Token
        Field: Token
        Example: 'README.txt' -> [1], 'README' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int('.' in form)]


# XXX Return is not bool
def token_long_pattern(token, _=None):
    """Convert token by it's casing pattern

    Args:
       token (str): The token
       _: Unused

    Returns:
       [Str]: For each letter uppercase -> A, lowercase -> a, not letter -> _

    HunTag:
        Type: Token
        Field: Token
        Example: 'README.txt' -> [AAAAA_aaa], 'README' -> [AAAAA]
        Use case: NER
    """
    pattern = ''
    for char in token:
        if char in smallcase:
            pattern += 'a'
        elif char in bigcase:
            pattern += 'A'
        else:
            pattern += '_'

    return [pattern]


# XXX Return is not bool
def token_short_pattern(token, _=None):
    """Convert token by it's shortened casing pattern

    Args:
       token (str): The token
       _: Unused

    Returns:
       [Str]: For each letter uppercase -> A, lowercase -> a, not letter -> _
              consecutive identical letters shortened to one letter

    HunTag:
        Type: Token
        Field: Token
        Example: 'README.txt' -> [A_a], 'README' -> [A]
        Use case: NER
    """
    pattern = ''
    prev = ''
    for char in token:
        if char in smallcase:
            if prev != 'a':
                pattern += 'a'
                prev = 'a'
        elif char in bigcase:
            if prev != 'A':
                pattern += 'A'
                prev = 'A'
        else:
            if prev != '_':
                pattern += '_'
                prev = '_'

    return [pattern]


# XXX Return is not bool
def token_chunk_tag(chunk_tag, _=None):
    """Returns the field as it is. (getForm do the same for non-merged tokens)

    Args:
       chunk_tag (str):  NP chunking tag
       _: Unused

    Returns:
       [Str]: The field as it is

    HunTag:
        Type: Token
        Field: Any
        Example: 'B-NP' -> [B-NP]
        Use case: NER

    Replaces:
        token_getBNCtag: is same
    """
    return [chunk_tag]


# XXX Return is not bool
def token_chunk_type(chunk_tag, _=None):
    """Returns the field type from the 3rd character

    Args:
       chunk_tag (str):  NP chunking tag
       _: Unused

    Returns:
       [Str]: The field from the 3rd character

    HunTag:
        Type: Token
        Field: Any
        Example: 'B-NP' -> [NP]
        Use case: NER, but not in SzegedNER

    Replaces:
        token_getTagType: is same
    """
    return [chunk_tag[2:]]


# XXX Return is not bool
def token_get_form(token, _=None):
    """Returns input if it has no underscore (_) in it, else returns 'MERGED'
     (Recski merged multi-token names by underscore (_) in chunking)

    Args:
       token (str): The token
       _: Unused

    Returns:
       [Str]: Token or MERGED if multi-token name merged by underscore

    HunTag:
        Type: Token
        Field: Token
        Example: 'Alma_fa' -> [MERGED], 'alma' -> [alma]
        Use case: NER, Chunk
    """
    if '_' not in token:
        return [token]
    return ['MERGED']


# XXX Return is not bool
def token_ngrams(token, options):
    """Make character n-grams

    Args:
       token (str): The token
       options (dict): n, the order of grams

    Returns:
       [Str]: List of Token's character n-grams

    HunTag:
        Type: Token
        Field: Token
        Example: 'alma', n=3 -> ['@alm', 'lma@'], 'almafa' -> ['@alm', 'lma', 'maf', 'afa@'], 'Y2K' -> ['@Y2K']
        Use case: NER, Chunk
    """
    n = int(options['n'])
    f = [str(token[c:c + n]) for c in range(max(0, len(token) - n + 1))]
    flen = len(f)
    if flen > 0:
        f[0] = '@{0}'.format(f[0])
        if flen > 1:
            f[-1] = '{0}@'.format(f[-1])
    return f


# XXX Return is not bool
def token_first_char(token, _=None):
    """Return the first character

    Args:
       token (str): The token
       _: Unused

    Returns:
       [Str]: Field's first character

    HunTag:
        Type: Token
        Field: Any field
        Example: 'B-NP' -> [B]
        Use case: NER, Chunk

    Replaces:
         token_chunkPart: is same
         token_msd_pos(?): is same
         token_posStart: is same
    """
    return [token[0]]


# XXX Return is not bool
def token_msd_pos(msd_anal, _=None):
    """Return the second character (Square brackets enclosed)

    Args:
       msd_anal (str): MSD code analysis
       _: Unused

    Returns:
       [Str]: Field's second character

    HunTag:
        Type: Token
        Field: Analysis
        Example: '[Nc-sa—s3]' -> N
        Use case: NER, Chunk
    """
    return [msd_anal[1]]


# XXX Return is not bool
def token_msd_pos_and_char(msd_anal, _=None):
    """MSD code's 'krPieces' function (Square brackets enclosed)

    Args:
       msd_anal (str):  MSD code analysis
       _: Unused

    Returns:
       [Str]: MSD code's pieces

    HunTag:
        Type: Token
        Field: Analysis
        Example: '[Nc-sa—s3]' -> [N2c, N]
        Use case: NER, Chunk
    """
    pos = msd_anal[1]  # main POS
    return ['{0}{1}{2}'.format(pos, c, ch) for c, ch in enumerate(msd_anal[2:-1]) if ch != '-']


# XXX Return is not bool
def token_prefix(token, options):
    """Make n-long prefix

    Args:
       token (str): The token
       options (dict): n, the length of prefix

    Returns:
       [Str]: An n-long prefix of the token

    HunTag:
        Type: Token
        Field: Token
        Example: 'alma', n=3 -> [alm], 'almafa' n=5 -> [almaf]
        Use case: NER
    """
    return [token[0:int(options['n'])]]


# XXX Return is not bool
def token_suffix(token, options):
    """Make n-long suffix

    Args:
       token (str): The token
       options (dict): n, the length of suffix

    Returns:
       [Str]: An n-long suffix of the token

    HunTag:
        Type: Token
        Field: Token
        Example: 'alma', n=3 -> [lma], 'almafa' n=5 -> [lmafa]
        Use case: NER
    """
    return [token[-int(options['n']):]]


# XXX Return is not bool
def sentence_lemma_lowered(sen, fields, _=None):
    """Lemma or Token has first letter capitalized

    Args:
       sen (list): List of tokens in the sentence
       fields (list): number of fields used (order: token, lemma)
       _: Unused

    Returns:
       [[Str/Int]]: 1,0,'raised' See the truth table below

    HunTag:
        Type: Sentence
        Field: Token, Lemma
        Example: 'alma', 'Alma -> ['raised'], 'Almafa', 'almafa' -> [1]
        Use case: NER
    """
    assert len(fields) == 2
    feat_vec = []
    for token, lemma in zip((tok[fields[0]] for tok in sen), (tok[fields[1]] for tok in sen)):
        if token[0] not in bigcase and big2small[lemma[0]] == token[0]:  # token lower and lemma upper
            feat_vec.append(['raised'])

        elif token[0] in bigcase and token[0] == lemma[0]:  # token upper and lemma upper
            feat_vec.append([0])

        elif token[0] in bigcase and big2small[token[0]] == lemma[0]:  # token upper and lemma lower
            feat_vec.append([1])

        feat_vec.append(['N/A'])  # token lower and lemma lower

    return feat_vec


# XXX Return is not bool
def token_kr_pieces(kr_anal, _=None):
    """Split KR code analysis to pieces

    Args:
       kr_anal (str): KR code analysis
       _: Unused

    Returns:
       [Str]: Pass KR code pieces

    HunTag:
        Type: Token
        Field: Token
        Example: ???
        Use case: NER, Chunk

    Known bug: token_kr_pieces is incorrect e.g. not all occurences of PLUR refer
               to NOUN. KR codes must be parsed in a more sophisticated manner
               -- we have the code that does so, but we must decide on what
               kr features should be like!
               see: https://github.com/recski/HunTag/issues/4
    """
    pieces = re.split(r'\W+', kr_anal.split('/')[-1])
    pos = pieces[0]
    feats = []
    last = ''
    for piece in pieces:
        if piece == 'PLUR':
            processed = '{0}_PLUR'.format(pos)
        elif piece in ('1', '2') or last == 'CAS':
            processed = '{0}_{1}'.format(last, piece)
        else:
            processed = piece
        if processed != 'CAS':
            feats.append(processed)

        last = piece

    return [feat for feat in feats if feat]


# XXX Return is not bool
def token_univ_pieces(univ_anal, _=None):
    """Split Univmorf feature-value pairs to pieces

    Args:
       univ_anal (str): Univmorf feature-value pairs
       _: Unused

    Returns:
       [Str]: Pass univmorf feature-value pairs

    HunTag:
        Type: Token
        Field: Token
        Example: 'Case=Nom|Number=Sing' --> ['Case=Nom', 'Number=Sing']
        Use case: Chunk

    """
    pieces = univ_anal.split('|')
    return [piece for piece in pieces if piece]


# XXX Return is not bool
def token_hfst_pieces(hfst_anal, _=None):
    """Split the morphological codes of the HFST-based eM-morph morphological analyser to pieces

    Args:
       hfst_anal (str): eM-morph morphological codes
       _: Unused

    Returns:
       [Str]: Pass eM-morph morphological code pieces

    HunTag:
        Type: Token
        Field: Token
        Example: '[Pl.Poss.3Sg][Nom]' --> ['Pl.Poss.3Sg', 'Nom']
        Use case: Chunk

    """
    pieces = hfst_anal.split('[')
    return [piece.strip('[]') for piece in pieces if piece]


# XXX Return is not bool
def token_full_kr_pieces(kr_anal, _=None):
    """Split KR code analysis to pieces from full analysis (with lemma)

    Args:
       kr_anal (str): KR code analysis
       _: Unused

    Returns:
       [Str]: Pass KR code pieces

    HunTag:
        Type: Token
        Field: Token
        Example: ???
        Use case: NER, Chunk
    """
    return token_kr_pieces('/'.join(kr_anal.split('/')[1:]))


def sentence_is_between_same_cases(sen, fields, options=None):
    """Is between same grammatical cases

    Args:
       sen (list): The list of tokens in the sentence
       fields (list): Field numbers, that will be used
       options (dict): options (max_dist default: 6

    Returns:
       [[Bool in int format]]: Pass the resulting array

    HunTag:
        Type: Sentence
        Field: Analysis
        Example: ???
        Use case: NER, Chunk
    """
    if options is None:
        options = {'max_dist': '6'}
    if len(fields) > 1:
        print('Error: "isBetweenSameCases" function\'s "fields" argument\'s\
            length must be one not {0}'.format(len(fields)), file=sys.stderr, flush=True)
        sys.exit(1)
    max_dist = int(options['max_dist'])
    noun_cases = [[] for _ in sen]
    feat_vec = [[] for _ in sen]
    kr_vec = [token[fields[0]] for token in sen]

    for c, kr in enumerate(kr_vec):
        if 'CAS' in kr:
            cases = re.findall(r'CAS<...>', kr)
            if not cases:
                noun_cases[c] = ['NO_CASE']
            else:
                case = cases[0][-4:-1]
                noun_cases[c] = [case]
        elif 'Case=' in kr:
            cases = re.findall(r'Case=[A-Z][a-z][a-z]', kr)
            if not cases:
                noun_cases[c] = ['NO_CASE']
            else:
                case = cases[0][-4:-1]
                noun_cases[c] = [case]

    left_case = {}
    right_case = {}
    curr_case = None
    case_pos = None
    for j, _ in enumerate(sen):
        if not noun_cases[j]:
            left_case[j] = (curr_case, case_pos)
        else:
            curr_case = noun_cases[j]
            case_pos = j
            left_case[j] = (None, None)

    curr_case = None
    case_pos = None
    for j in range(len(sen) - 1, -1, -1):
        if not noun_cases[j]:
            right_case[j] = (curr_case, case_pos)
        else:
            curr_case = noun_cases[j]
            case_pos = j
            right_case[j] = (None, None)

    for j, _ in enumerate(sen):
        feat_vec[j] = [0]
        if (right_case[j][0] == left_case[j][0] and right_case[j][0] is not None and
                abs(right_case[j][1] - left_case[j][1]) <= max_dist):
            feat_vec[j] = [1]

    return feat_vec


# XXX Return is not bool
def token_get_pos_tag(kr_anal, _=None):
    """Return KR code POS tag

    Args:
       kr_anal (str): KR code analysis
       _: Unused

    Returns:
       [Str]: Pass ???

    HunTag:
        Type: Token
        Field: Analysis
        Example: ???
        Use case: NER, Chunk
    """
    return [re.split(r'\W+', kr_anal.split('/')[-1])[0]]


# XXX Return is not bool
def sentence_kr_patts(sen, fields, options):
    """Return KR code patterns
    G. Recski, D. Varga: Magyar főnévi csoportok azonosítása In: Általános Nyelvészeti Tanulmányok 24. 2012.
    http://people.mokk.bme.hu/~recski/pub/huntag_anyt.pdf Page 6
    G. Recski MA thesis: NP chunking in Hungarian
    http://people.mokk.bme.hu/~recski/pub/np_thesis.pdf Page 21

    Args:
       sen (list): List of tokens in the sentence
       fields (list): number of fields used
       options (dict): available options

    Returns:
       [[Str]]: Pass KR code patterns (see above)

    HunTag:
        Type: Sentence
        Field: Analysis
        Example: ???
        Use case: NER, Chunk

    Replaces:
       parsePatts: superseded
       mySpecPatts: superseded
       myPatts: Unimplementable, almost same
       sentence_parsePatts: superseded
    """
    assert options['lang'] in ('en', 'hu')
    min_length = int(options['min_length'])
    max_length = int(options['max_length'])
    rad = int(options['rad'])
    assert len(fields) == 1
    f = fields[0]
    feat_vec = [[] for _ in sen]
    kr_vec = [tok[f] for tok in sen]

    if options['lang'] == 'hu':
        if not options['full_kr'] and not options['msd']:
            kr_vec = [token_get_pos_tag(kr)[0] for kr in kr_vec]
        elif options['msd']:
            kr_vec = [tok[f] for tok in sen]
    else:
        kr_vec = [tok[f][0] for tok in sen]

    apply_cas_diff_fun = do_nothing
    apply_poss_connect_fun = do_nothing

    if options['lang'] == 'hu':
        if options['msd']:
            tag_dt, feat_prefix_dt = '[Tf]', 'dt_'  # "(since) last detrminant" MSD
            cas_re, feat_name = cas_re_msd, 'cas_diff'  # cas_diff
            poss_re, obj, feat_prefix_poss = possessor_msd, obj_msd, 'possession_'
        else:
            tag_dt, feat_prefix_dt = 'DT', 'dt_'    # "(since) last detrminant" KR
            cas_re, feat_name = cas_re_kr, 'cas_diff'   # cas_diff
            poss_re, obj, feat_prefix_poss = possessor_kr, obj_kr, 'possession_'
        if options['cas_diff'] == 1:
            apply_cas_diff_fun = cas_diff
        if options['poss_connect'] == 1:
            apply_poss_connect_fun = poss_connect
    else:
        tag_dt, feat_prefix_dt = 'DT', 'dt_'  # "(since) last detrminant" CoNLL (poss_connect and CasDiff not used)
        cas_re, feat_name = None, None
        poss_re, obj, feat_prefix_poss = None, None, None

    if options['since_dt'] == 1:
        apply_since_pos_fun = since_pos
    else:
        apply_since_pos_fun = do_nothing

    assert len(kr_vec) == len(sen)
    kr_vec_len = len(kr_vec)
    # For every token in sentence
    for c in range(kr_vec_len):
        apply_since_pos_fun(kr_vec, c, feat_vec[c], tag_dt, feat_prefix_dt)
        apply_cas_diff_fun(kr_vec, c, feat_vec[c], cas_re, feat_name)
        apply_poss_connect_fun(kr_vec, c, feat_vec[c], poss_re, obj, feat_prefix_poss)
        # Begining in -rad and rad but starts in the list boundaries (lower)
        for k in range(max(-rad, -c), rad):
            # Ending in -rad + 1 and rad + 2  but starts in the list boundaries (upper)
            # and keep minimal and maximal length
            for j in range(max(-rad + 1, min_length + k), min(rad + 2, max_length + k + 1, kr_vec_len - c + 1)):
                value = '+'.join(kr_vec[c + k:c + j])
                feat = '{0}_{1}_{2}'.format(k, j, value)
                feat_vec[c].append(feat)
    return feat_vec


def token_kr_plural(kr_anal, _=None):
    """Detect plural form in KR code

    Args:
       kr_anal (str): KR code analysis
       _: Unused

    Returns:
       [Bool in int format]: True if KR code is plural

    HunTag:
        Type: Token
        Field: Analysis
        Example: ???
        Use case: NER, Chunk
    """
    return [int('NOUN<PLUR' in kr_anal)]


def token_univ_plural(univ_anal, _=None):
    """Detect plural form in univmorf feature-value pairs

    Args:
       univ_anal (str): univmorf code analysis
       _: Unused

    Returns:
       [Bool in int format]: True if univmorf code contains plural

    HunTag:
        Type: Token
        Field: Analysis
        Example: 'teendők' Case=Nom|Number=Plur --> univPlural = 1
        Use case: NER, Chunk
    """
    return [int('Number=Plural' in univ_anal)]


# XXX Return is not bool
def token_hfst_plural(hfst_anal, _=None):
    """Detect plural form in the new formalism of HFST-based Hungarian morphological analyser (aka eM-morph)

    Args:
       hfst_anal (str): hfstmorf code analysis
       _: Unused

    Returns:
       [Bool in int format]: True if hfst code contains plural

    HunTag:
        Type: Token
        Field: Analysis
        Example: 'teendők' [Pl][Nom] --> hfstPlural = 1
        Use case: NER
    """
    return [int('[Pl]' in hfst_anal)]


# XXX Return is not bool
def token_get_np_part(chunk_tag, _=None):
    """Checks if the token is part of NP

    Args:
       chunk_tag (Str): NP chunking tag
       _: Unused

    Returns:
       [Str]: Return tag's first character or 'O'

    HunTag:
        Type: Token
        Field: NP chunks
        Example: ???
        Use case: NER, Chunk
    """
    if chunk_tag == 'O' or chunk_tag[2:] != 'NP':
        return ['O']
    else:
        return [chunk_tag[0]]


def token_cap_period_operator(form, _=None):
    """Token is an uppercase letter followed by a period (from Bikel et al. (1999))

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: Matches re [A-Z]\.$

    HunTag:
        Type: Token
        Field: Token
        Example: 'A.' -> [1], 'alma' -> [0]
        Use case: NER
    """
    return [int(bool(re.match(r'[A-Z]\.$', form)))]


def token_is_digit_operator(form, _=None):
    """Token is number

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Token is number

    HunTag:
        Type: Token
        Field: Token
        Example: '333' -> [1], '3-gram' -> [0]
        Use case: NER
    """
    return [int(form.isdigit())]


def token_one_digit_num_operator(form, _=None):
    """Token is one digit (from Zhou and Su (2002))

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Token is one digit

    HunTag:
        Type: Token
        Field: Token
        Example: '3' -> [1], '333' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int(len(form) == 1 and form.isdigit())]


def token_two_digit_num_operator(form, _=None):
    """Token is two digit (from Bikel et al. (1999))

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Token is two digit

    HunTag:
        Type: Token
        Field: Token
        Example: '33' -> [1], '333' -> [0]
        Use case: NER
    """
    return [int(len(form) == 2 and form.isdigit())]


def token_three_digit_num_operator(form, _=None):
    """Token is three digit

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Token is three digit

    HunTag:
        Type: Token
        Field: Token
        Example: '333' -> [1], '3333' -> [0]
        Use case: NER
    """
    return [int(len(form) == 3 and form.isdigit())]


def token_four_digit_num_operator(form, _=None):
    """Token is four digit

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Token is four digit

    HunTag:
        Type: Token
        Field: Token
        Example: '2015' -> [1], '333' -> [0]
        Use case: NER
    """
    return [int(len(form) == 4 and form.isdigit())]


def token_is_punctuation_operator(form, _=None):
    """Token is punctuation

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Token is punctuation

    HunTag:
        Type: Token
        Field: Token
        Example: '.' -> [1], 'A.' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int(set(form).issubset(set(',.!"\'():?<>[];{}')))]


def token_contains_digit_and_dash_operator(form, _=None):
    """Token contains digit and hyphen (-) (from Bikel et al. (1999))

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Token contains digits and hyphen (-)

    HunTag:
        Type: Token
        Field: Token
        Example: '2014-15' -> [1], '3-gram' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int(bool(re.match('[0-9]+-[0-9]+', form)))]


def token_contains_digit_and_slash_operator(form, _=None):
    """Token contains digit and slash (/) (from Bikel et al. (1999))

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Token contains digit and slash (/)

    HunTag:
        Type: Token
        Field: Token
        Example: '2014/2015' -> [1], '3/A' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int(bool(re.match('[0-9]+/[0-9]+', form)))]


def token_contains_digit_and_comma_operator(form, _=None):
    """Token contains digit and comma (. or ,) (from Bikel et al. (1999))

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Token contains digit and comma (. or ,)

    HunTag:
        Type: Token
        Field: Token
        Example: '2015.04.07.' -> [1], '2015.txt' -> [0]
        Use case: NER
    """
    return [int(bool(re.match('[0-9]+[,.][0-9]+', form)))]


def token_year_decade_operator(form, _=None):
    """Token contains year decade (from Zhou and Su (2002))

    Args:
       form (str): The token
       _: Unused

    Returns:
       [Bool in int format]: True if Token contains year decade

    HunTag:
        Type: Token
        Field: Token
        Example: '1990s' -> [1], '80s' -> [1]
        Use case: NER
    """
    return [int(bool(re.match('[0-9][0-9]s$', form) or
                re.match('[0-9][0-9][0-9][0-9]s$', form)))]


def sentence_new_sentence_start(sen, *_):
    """Sentence starting token or not

    Args:
       sen (list): List of tokens in the sentence
       _: Unused

    Returns:
       [[Bool in int format]]: True if Token is at the sentence start

    HunTag:
        Type: Sentence
        Field: Token
        Example: ???
        Use case: NER, Chunk
    """
    feat_vec = [[0] for _ in sen]
    feat_vec[0][0] = 1
    return feat_vec


def sentence_new_sentence_end(sen, *_):
    """Sentence ending token or not

    Args:
       sen (list): List of tokens in the sentence
       _: Unused

    Returns:
       [[Bool in int format]]: True if Token is at the sentence end

    HunTag:
        Type: Sentence
        Field: Token
        Example: ???
        Use case: NER, Chunk
    """
    feat_vec = [[0] for _ in sen]
    feat_vec[-1][0] = 1
    return feat_vec


def token_unknown(anal, _=None):
    """Guessed Unknown word from HunMorph?

    Args:
       anal (str): The analysis
       _: Unused

    Returns:
       [[Bool in int format]]: True if Tokens analysis contains OOV OR UNKNOWN

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: NER

    Replaces:
        token_OOV: superseded
    """
    return [int('UNKNOWN' in anal or 'OOV' in anal)]


# XXX Return is not bool
def token_get_kr_lemma(kr_lemma, _=None):
    """Get lemma from HunMorph's analysis

    Args:
       kr_lemma (str): The token
       _: Unused

    Returns:
       [Str]: Returns lemma

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: NER
    """
    return [kr_lemma.split('/')[0]]


# XXX Return is not bool
def token_get_kr_pos(kr_anal, _=None):
    """Get KR code POS tag

    Args:
       kr_anal (str): The token
       _: Unused

    Returns:
       [Str]: Returns POS part of KR code

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: NER, but not in SzegedNER
    """
    if '<' in kr_anal:
        return [kr_anal.split('<')[0]]
    return [kr_anal]


# XXX Return is not bool
def token_get_penn_tags(penn_tag, _=None):
    """Reduces Penn tagset's similar tags

    Args:
       penn_tag (str): Penn Tag
       _: Unused

    Returns:
       [Str]: ???

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: NER
    """
    if re.match('^N', penn_tag) or re.match('^PRP', penn_tag):
        return ['noun']
    elif penn_tag == 'IN' or penn_tag == 'TO' or penn_tag == 'RP':
        return ['prep']
    elif re.match('DT$', penn_tag):
        return ['det']
    elif re.match('^VB', penn_tag) or penn_tag == 'MD':
        return ['verb']
    else:
        return ['0']


def token_humor_plural(humor_tag, _=None):
    """Check if Humor code plural

    Args:
       humor_tag (str): Humor analysis
       _: Unused

    Returns:
       [[Bool in int format]]: True if Humor analysis is plural

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: NER
    """
    return [int('PL' in humor_tag)]


# XXX Return is not bool
def token_get_kr_end(kr_anal, _=None):
    """Return KR code end

    Args:
       kr_anal (str): The token
       _: Unused

    Returns:
       [Str]: ???

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: NER, but not in SzegedNER
    """
    end = '0'
    if '<' in kr_anal:
        pieces = kr_anal.split('<', 1)
        end = pieces[1]

    return [end]


def token_penn_plural(penn_tag, _=None):
    """Check if Penn code plural

    Args:
       penn_tag (str): Penn tag
       _: Unused

    Returns:
       [Bool in int format]: ???

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: NER, but not in SzegedNER

    Replaces:
        token_plural: is same
    """
    return [int(penn_tag == 'NNS' or penn_tag == 'NNPS')]


# XXX Return is not bool
def token_humor_pieces(humor_tag, _=None):
    """Return Humor tag pieces
    For Humor code:
    'FN/N|PROP|FIRST/ffinev;veznev' => ['FN', 'N|PROP|FIRST', 'ffinev;veznev']
    'FN+PSe3+DEL' => ['FN', 'PSe3', 'DEL']

    Args:
       humor_tag (str): Humor tag
       _: Unused

    Returns:
       [str]: ???

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: Chunk, NER
    """
    return [item for part in humor_tag.split('/') for item in part.split('+')]


# XXX Return is not bool
def token_humor_simple(humor_tag, _=None):
    """Humor test: return self

    Args:
       humor_tag (str): Humor tag
       _: Unused

    Returns:
       [str]: self

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: Chunk, NER
    """
    return [humor_tag]


# XXX Return is not bool
def token_wordnet_simple(word_net_tags, _=None):
    """Wordnet synsets as tags

    Args:
       word_net_tags (str): wordNet synsets as tags
       _: Unused

    Returns:
       [str]: list of wordNet synsets as features

    HunTag:
        Type: Token
        Field: WordNet tags
        Example: animate.n.1/human.n.1 -> ['animate.n.1', 'human.n.1'], '' -> []
        Use case: Chunk, NER
    """
    if len(word_net_tags) == 0:
        return []
    return word_net_tags.split('/')


# XXX Return is not bool
def token_mmo_simple(mmo_tags, _=None):
    """MMO properties

    Args:
       mmo_tags (str): wordNet synsets as tags
       _: Unused

    Returns:
       [str]: list of MMO tags as features

    HunTag:
        Type: Token
        Field: MMOtags
        Example:
            in:'NX[abstract=YES,animate=NIL,auth=YES,company=NIL,encnt=YES,human=NIL]'
            out: ['NX', 'abstract=YES', 'animate=NIL', 'auth=YES', 'company=NIL', 'encnt=YES', 'human=NIL']
        Use case: Chunk, NER
    """
    if mmo_tags == '-':
        return []
    return [x for x in mmo_patt.split(mmo_tags) if x]
