#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-
"""
features.py stores implementations of individual feature types for use with
HunTag. Feel free to add your own features but please add comments to describe
them.
"""

import re
import sys
from copy import deepcopy


# XXX Return is not bool
# XXX and not list (Fixed)
def stupidStem(token):
    """Stem tokens with hyphen (-) in them.

    Args:
       token (str): The token

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


### WILL BE DELETED
# The original isCapitalizedOperator, but this checks only
# if it has uppercase in it anywhere
def isCapitalizedOperator(form):
    return [int(form.lower() != form)]


def hasCapOperator(form):
    """Has it capital letter anywhere?

    Args:
       form (str): The token

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


### WILL BE DELETED
# New: all lowercase
def lowerCaseOperator(form):
    return [int(form.lower() == form)]


def isCapOperator(form):
    """Only the first letter is capital
    This is the new isCapitalized: Starts with uppercase

    Args:
       form (str): The token

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


### WILL BE DELETED
# This is new: Starts lowercase
def notCapitalizedOperator(form):
    return [int(form[0] == form[0].lower())]


def isAllcapsOperator(form):
    """StupidStem consists of uppercase letters

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if token_stupidStem is uppercase

    HunTag:
        Type: Token
        Field: Token
        Example: 'MTI-vel' -> [1], 'Mti-vel' -> [0]
        Use case: NER
    """
    return [int(stupidStem(form)[0].isupper())]


# XXX token_stupidStem or whole token?
def isCamelOperator(form):
    """The first letter is lower, the others has, but not all uppercase (camelCasing)

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if Not all letter are uppercase, but at least one is from the second character

    HunTag:
        Type: Token
        Field: Token
        Example: 'aLMa' -> [1], 'ALMA' -> [0], 'alma' -> [0]
        Use case: NER
    """
    return [int(not isAllcapsOperator(form) and form[1:].lower() != form[1:])]


# XXX token_stupidStem or whole token?
def threeCaps(form):
    """Token is three uppercase letters?

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if Token consist of three uppercase letters

    HunTag:
        Type: Token
        Field: Token
        Example: 'MTI' -> [1], 'Mti' -> [0], 'Matáv' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int(len(form) == 3 and stupidStem(form)[0].isupper())]


def startsWithNumberOperator(form):
    """Token starts with number

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if Token's first letter is a digit

    HunTag:
        Type: Token
        Field: Token
        Example: '3-gram' -> [1], 'n-gram' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int(form[0].isdigit())]


### WILL BE DELETED
def isNumberOperator(form):
    """stupidStem contains letters or just numbers and punctuation?

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if token_stupidStem is only numbers or punctuation

    HunTag:
        Type: Token
        Field: Token
        Example: '3-gram' -> [1], 'n-gram' -> [0]
        Use case: NER
    """
    return [int(set(stupidStem(form)[0]).issubset(set('0123456789,.-%')))]


def hasNumberOperator(form):
    """Token contains numbers

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if token contains numbers

    HunTag:
        Type: Token
        Field: Token
        Example: '3-gram' -> [1], 'n-gram' -> [0]
        Use case: NER
    """
    return [int(not set('0123456789').isdisjoint(set(form)))]


def hasDashOperator(form):
    """Token contains hyphen (-)

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if token contains hyphen (-)

    HunTag:
        Type: Token
        Field: Token
        Example: '3-gram' -> [1], 'ngram' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int('-' in form)]


def hasUnderscoreOperator(form):
    """Token contains underscore (_)

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if token contains underscore (_)

    HunTag:
        Type: Token
        Field: Token
        Example: 'function_name' -> [1], 'functionName' -> [0]
        Use case: Chunk
    """
    return [int('_' in form)]


def hasPeriodOperator(form):
    """Token contains period (.)

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if token contains period (.)

    HunTag:
        Type: Token
        Field: Token
        Example: 'README.txt' -> [1], 'README' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int('.' in form)]


### WILL BE DELETED
def sentenceStart(sen):
    featureVector = []
    for formIndex, form in enumerate(sen):
        if form == '':
            featureVector.append('')
        else:
            if formIndex > 0:
                val = int(sen[formIndex - 1] == '')
            else:
                val = 1
            featureVector.append(val)
    return featureVector


### WILL BE DELETED
def sentenceEnd(sen):
    featureVector = []
    for formIndex, form in enumerate(sen):
        if form == '':
            featureVector.append('')
        else:
            if formIndex + 1 < len(sen):
                val = int(sen[formIndex + 1] == '')
            else:
                val = 1
            featureVector.append(val)
    return featureVector


# GLOBAL DECLARATION BEGIN
# see: token_longPattern, token_shortPattern, token_scase, token_lemmaLowered
smallcase = 'aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz'
bigcase = 'AÁBCDEÉFGHIÍJKLMNOÓÖŐPQRSTUÚÜŰVWXYZ'
big2small = {}
for i, _ in enumerate(bigcase):
    big2small[bigcase[i]] = smallcase[i]
# GLOBAL DECLARATION END


# XXX Return is not bool
def longPattern(token):
    """Convert token by it's casing pattern

    Args:
       token (str): The token

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
def shortPattern(token):
    """Convert token by it's shortened casing pattern

    Args:
       token (str): The token

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


### WILL BE DELETED
def scase(token):
    """Lowercase token's first letter

    Args:
       token (str): The token

    Returns:
       [Str]: Token, first letter lowercased

    HunTag:
        Type: Token
        Field: Token
        Example: 'Alma' -> [alma], 'ALMA' -> [aLMA]
        Use case: NER
    """
    if token[0] in bigcase:
        token = token[0].lower() + token[1:]

    return [token]

"""
### WILL BE DELETED
def isInRangeWithSmallCase(word):
    # print(word, file=sys.stderr, flush=True)
    checkedRange = 30
    if word[0] in smallcase:
        return 'n/a'
    else:
        return [int(db.isInRange(scase(word), checkedRange, wordcount))]
"""


# XXX Return is not bool
def fchunkTag(chunkTag):
    """Returns the field as it is. (getForm do the same for non-merged tokens)

    Args:
       chunkTag (str):  NP chunking tag

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
    return [chunkTag]


# XXX Return is not bool
def chunkType(chunkTag):
    """Returns the field type from the 3rd character

    Args:
       chunkTag (str):  NP chunking tag

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
    return [chunkTag[2:]]


### WILL BE DELETED
def chunkPart(chunktag):
    # 'B-NP' -> B
    return [chunktag[0]]


# XXX Return is not bool
def getForm(token):
    """Returns input if it has no underscore (_) in it, else returns 'MERGED'
     (Recski merged multi-token names by underscore (_) in chunking)

    Args:
       token (str): The token

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
def ngrams(token, options):
    """Make character n-grams

    Args:
       token (str): The token
       options (dict): n, the order of grams

    Returns:
       [Str]: List of Token's character n-grams

    HunTag:
        Type: Token
        Field: Token
        Example: 'alma', n=3 -> [@alm, lma], 'almafa' -> [@alm, lma, maf, afa@]
        Use case: NER, Chunk
    """
    n = int(options['n'])
    f = [str(token[c:c + n]) for c in range(max(0, len(token) - n + 1))]
    if len(f) > 0:
        f[0] = '@{0}'.format(f[0])
        f[-1] = '{0}@'.format(f[-1])
    return f


# XXX Return is not bool
def firstChar(token):
    """Return the first character

    Args:
       token (str): The token

    Returns:
       [Str]: Field's first character

    HunTag:
        Type: Token
        Field: Any field
        Example: 'B-NP' -> [B]
        Use case: NER, Chunk

    Replaces:
         token_chunkPart: is same
         token_msdPos(?): is same
         token_posStart: is same
    """
    return [token[0]]


# XXX Return is not bool
# XXX and not list (Fixed)
def msdPos(msdAnal):
    """Return the second character (Square brackets enclosed)

    Args:
       msdAnal (str): MSD code analysis

    Returns:
       [Str]: Field's second character

    HunTag:
        Type: Token
        Field: Analysis
        Example: '[Nc-sa—s3]' -> N
        Use case: NER, Chunk
    """
    return [msdAnal[1]]


# XXX Return is not bool
def msdPosAndChar(msdAnal):
    """MSD code's 'krPieces' function (Square brackets enclosed)

    Args:
       msdAnal (str):  MSD code analysis

    Returns:
       [Str]: MSD code's pieces

    HunTag:
        Type: Token
        Field: Analysis
        Example: '[Nc-sa—s3]' -> [N2c, N]
        Use case: NER, Chunk
    """
    pos = msdAnal[1]  # main POS
    return ['{0}{1}{2}'.format(pos, c, ch) for c, ch in enumerate(msdAnal[2:-1]) if ch != '-']


# XXX Return is not bool
def prefix(token, options):
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
def suffix(token, options):
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
# XXX Arguments potentially wrong
def lemmaLowered(sen, fields):
    """Lemma or Token has first letter capitalized

    Args:
       sen (list): List of tokens in the sentence
       fields (list): number of fields used (order: token, lemma)

    Returns:
       [[Str/Int]]: 1,0,'raised' See the truth table below

    HunTag:
        Type: Sentence
        Field: Token, Lemma
        Example: 'alma', 'Alma -> ['raised'], 'Almafa', 'almafa' -> [1]
        Use case: NER
    """
    assert len(fields) == 2
    featVec = []
    for token, lemma in zip((tok[fields[0]] for tok in sen), (tok[fields[1]] for tok in sen)):
        if token[0] not in bigcase and big2small[lemma[0]] == token[0]:  # token lower and lemma upper
            featVec.append(['raised'])

        elif token[0] in bigcase and token[0] == lemma[0]:  # token upper and lemma upper
            featVec.append([0])

        elif token[0] in bigcase and big2small[token[0]] == lemma[0]:  # token upper and lemma lower
            featVec.append([1])

        featVec.append(['N/A'])  # token lower and lemma lower

    return featVec


# XXX Return is not bool
def krPieces(krAnal):
    """Split KR code analysis to pieces

    Args:
       krAnal (str): KR code analysis

    Returns:
       [Str]: Pass KR code pieces

    HunTag:
        Type: Token
        Field: Token
        Example: ???
        Use case: NER, Chunk

    Known bug: token_krPieces is incorrect e.g. not all occurences of PLUR refer
               to NOUN. KR codes must be parsed in a more sophisticated manner
               -- we have the code that does so, but we must decide on what
               kr features should be like!
               see: https://github.com/recski/HunTag/issues/4
    """
    pieces = re.split(r'\W+', krAnal.split('/')[-1])
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
def fullKrPieces(krAnal):
    """Split KR code analysis to pieces from full analysis (with lemma)

    Args:
       krAnal (str): KR code analysis

    Returns:
       [Str]: Pass KR code pieces

    HunTag:
        Type: Token
        Field: Token
        Example: ???
        Use case: NER, Chunk
    """
    return krPieces('/'.join(krAnal.split('/')[1:]))


### WILL BE DELETED
def krFeats(kr):
    pieces = re.split(r'\W+', kr)[1:]
    feats = []
    last = ''
    for piece in pieces:
        if piece in ('1', '2'):
            processed = '{0}_{1}'.format(last, piece)
        else:
            processed = piece

        feats.append(processed)
        last = piece

    return [feat for feat in feats if feat]


### WILL BE DELETED
def krConjs(kr):
    pieces = re.split(r'\W+', kr)
    conjs = []

    for ind, e1 in enumerate(pieces):
        for e2 in pieces[ind + 1:]:
            if e2 == '':
                continue
            conjs.append('{0}+{1}'.format(e1, e2))

    return [feat for feat in conjs if feat]


### WILL BE DELETED
def capsPattern(sen, fields):
    """Capitalized word sequence patterns"""
    featVec = [[] for _ in sen]

    assert len(fields) == 1
    tokens = [word[fields[0]] for word in sen]
    upperFlags = [hasCapOperator(token)[0] for token in tokens]
    start = -1
    mapStartToSize = {}
    for pos, flag in enumerate(upperFlags + [0]):
        if flag == 0:
            if start != -1:
                mapStartToSize[start] = pos - start
            start = -1
            continue
        else:
            if start == -1:
                start = pos
        if start != -1:
            mapStartToSize[start] = len(upperFlags) - start

    for pos, flag in enumerate(upperFlags):
        if flag == 0:
            start = -1
            continue
        if start == -1:
            start = pos
        positionInsideCapSeq = pos - start
        lengthOfCapSeq = mapStartToSize[start]
        p = str(positionInsideCapSeq)
        l = str(lengthOfCapSeq)
        featVec[pos] += ['p{0}'.format(p),
                         'l{0}'.format(l),
                         'p{0}l{1}'.format(p, l)]

    return featVec


### WILL BE DELETED
def capsPattern_test():
    tokens = 'A certain Ratio Of GDP is Gone Forever'.split()
    sentence = [[token] for token in tokens]
    fields = [0]
    print(capsPattern(sentence, fields))


def isBetweenSameCases(sen, fields, options=None):
    """Is between same grammatical cases

    Args:
       sen (list): The list of tokens in the sentence
       fields (list): Field numbers, that will be used
       options (dict): options (maxDist default: 6

    Returns:
       [[Bool in int format]]: Pass the resulting array

    HunTag:
        Type: Sentence
        Field: Analysis
        Example: ???
        Use case: NER, Chunk
    """
    if options is None:
        options = {'maxDist': '6'}
    if len(fields) > 1:
        print('Error: "isBetweenSameCases" function\'s "fields" argument\'s\
            length must be one not {0}'.format(len(fields)), file=sys.stderr, flush=True)
        sys.exit(1)
    maxDist = int(options['maxDist'])
    nounCases = [[] for _ in sen]
    featVec = [[] for _ in sen]
    krVec = [token[fields[0]] for token in sen]

    for c, kr in enumerate(krVec):
        if 'CAS' in kr:
            cases = re.findall(r'CAS<...>', kr)
            if not cases:
                nounCases[c] = ['NO_CASE']
            else:
                case = cases[0][-4:-1]
                nounCases[c] = [case]

    leftCase = {}
    rightCase = {}
    currCase = None
    casePos = None
    for j, _ in enumerate(sen):
        if not nounCases[j]:
            leftCase[j] = (currCase, casePos)
        else:
            currCase = nounCases[j]
            casePos = j
            leftCase[j] = (None, None)

    currCase = None
    casePos = None
    for j in range(len(sen) - 1, -1, -1):
        if not nounCases[j]:
            rightCase[j] = (currCase, casePos)
        else:
            currCase = nounCases[j]
            casePos = j
            rightCase[j] = (None, None)

    for j, _ in enumerate(sen):
        featVec[j] = [0]
        if (rightCase[j][0] == leftCase[j][0] and rightCase[j][0] is not None and
                abs(rightCase[j][1] - leftCase[j][1]) <= maxDist):
            featVec[j] = [1]

    return featVec


# XXX Return is not bool
def getPosTag(krAnal):
    """Return KR code POS tag

    Args:
       krAnal (str): KR code analysis

    Returns:
       [Str]: Pass ???

    HunTag:
        Type: Token
        Field: Analysis
        Example: ???
        Use case: NER, Chunk
    """
    return [re.split(r'\W+', krAnal.split('/')[-1])[0]]


### WILL BE DELETED
# XXX Return is not bool
# XXX and not list
def tags_since_dt(sentence, tokRange):
    """
    Last determinant feature:
    All the POS tags that occured since the last determinant is joined with '+'
    Determinant: XXX TODO: Make this a parameter
    English: 'DT'
    Hungarian (KR code): '[Tf]'

    Args:
       sentence (list): List of tokens in the sentence
       tokRange (int): number of tokens used

    Returns:
       [Str]: Pass ???

    HunTag:
        Type: Sentence
        Field: Analysis
        Example: ???
        Use case: Chunk
    """
    tags = set()
    for pos in sentence[:tokRange]:
        if pos == '[Tf]':  # [Tf], DT
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))


def sincePos(krVec, c, tag, featPrefix, featVecElem):
    tagst = tags_since_pos(krVec, c, tag)
    if len(tagst) > 0:
        featVecElem.append(featPrefix + tagst)


def doNothing(*_):
    pass


# XXX LEFORDÍTANI
def casDiff(c, krVec, featVecElem):
    lastF = '' if c == 0 else krVec[c - 1]
    if lastF.startswith('[N') and krVec[c].startswith('[N') and lastF != krVec[c]:
        featVecElem.append('FNelter')


# XXX LEFORDÍTANI
def possConnect(c, krVec, featVecElem):
    if birtokos.search(krVec[c]):
        tagst = tags_since_pos(krVec, c, '^\[?N', False)
        if len(tagst) > 0:
            featVecElem.append('birtok_' + tagst)


# fullKr == True
# options['since_dt'] == 1
# options['CASDiff'] == 1
# options['POSSConnect'] == 1
# options['lang'] in ('en', 'hu')
# XXX Return is not bool
def krPatts(sen, fields, options, fullKr=False):
    """Return KR code patterns
        http://people.mokk.bme.hu/~recski/pub/huntag_anyt.pdf page 6

    Args:
       sen (list): List of tokens in the sentence
       fields (list): number of fields used
       options (dict): available options

    Returns:
       [[Str]]: Pass ???

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
    # options = {'lang': 'hu', 'since_dt': 1, 'CASDiff': 1, 'POSSConnect': 1}  # XXX TEST ONLY
    fullKr = True
    assert options['lang'] in ('en', 'hu')
    minLength = int(options['minLength'])
    maxLength = int(options['maxLength'])
    rad = int(options['rad'])
    assert len(fields) == 1
    f = fields[0]
    featVec = [[] for _ in sen]
    krVec = [tok[f] for tok in sen]

    if options['lang'] == 'hu':
        if not fullKr:
            krVec = [getPosTag(kr) for kr in krVec]
    else:
        krVec = [tok[f][0] for tok in sen]

    applyCasDiffFun = doNothing
    applyPossConnectFun = doNothing

    if options['lang'] == 'hu':
        tag, featPrefix = '[Tf]', 'dt_'
        if options['CASDiff'] == '1':
            applyCasDiffFun = casDiff
        if options['POSSConnect'] == '1':
            applyPossConnectFun = possConnect
    else:
        tag, featPrefix = 'DT', 'dt_'

    if options['since_dt'] == '1':
        applySincePosFun = sincePos
    else:
        applySincePosFun = doNothing

    assert len(krVec) == len(sen)
    krVecLen = len(krVec)
    # For every token in sentence
    for c in range(krVecLen):
        applySincePosFun(krVec, c, tag, featPrefix, featVec[c])
        applyCasDiffFun(c, krVec, featVec[c])
        applyPossConnectFun(c, krVec, featVec[c])
        # Begining in -rad and rad but starts in the list boundaries (lower)
        for k in range(max(-rad, -c), rad):
            # Ending in -rad + 1 and rad + 2  but starts in the list boundaries (upper)
            # and keep minimal and maximal length
            for j in range(max(-rad + 1, minLength + k), min(rad + 2, maxLength + k + 1, krVecLen - c + 1)):
                value = '+'.join(krVec[c + k:c + j])
                feat = '{0}_{1}_{2}'.format(k, j, value)
                featVec[c].append(feat)
    return featVec


def krPlural(krAnal):
    """Detect plural form in KR code

    Args:
       krAnal (str): KR code analysis

    Returns:
       [Bool in int format]: True if KR code is plural

    HunTag:
        Type: Token
        Field: Analysis
        Example: ???
        Use case: NER, Chunk
    """
    return [int('NOUN<PLUR' in krAnal)]


### WILL BE DELETED
def getTagType(tag):
    return tag[2:]


### WILL BE DELETED
def posStart(postag):
    return postag[0]


### WILL BE DELETED
def posEnd(postag):
    """Last character of field (BNC tag)"""
    return postag[-1]


# XXX Return is not bool
# XXX and not list (Fixed)
def getNpPart(chunkTag):
    """Checks if the token is part of NP

    Args:
       chunkTag (Str): NP chunking tag

    Returns:
       [Str]: Return tag's first character or 'O'

    HunTag:
        Type: Token
        Field: NP chunks
        Example: ???
        Use case: NER, Chunk
    """
    if chunkTag == 'O' or chunkTag[2:] != 'NP':
        return ['O']
    else:
        return [chunkTag[0]]


# XXX lowercased function name: CapPeriodOperator
def capPeriodOperator(form):
    """Token is an uppercase letter followed by a period (from Bikel et al. (1999))

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: Matches re [A-Z]\.$

    HunTag:
        Type: Token
        Field: Token
        Example: 'A.' -> [1], 'alma' -> [0]
        Use case: NER
    """
    return [int(bool(re.match(r'[A-Z]\.$', form)))]


def isDigitOperator(form):
    """Token is number

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if Token is number

    HunTag:
        Type: Token
        Field: Token
        Example: '333' -> [1], '3-gram' -> [0]
        Use case: NER
    """
    return [int(form.isdigit())]


def oneDigitNumOperator(form):
    """Token is one digit (from Zhou and Su (2002))

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if Token is one digit

    HunTag:
        Type: Token
        Field: Token
        Example: '3' -> [1], '333' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int(len(form) == 1 and form.isdigit())]


def twoDigitNumOperator(form):
    """Token is two digit (from Bikel et al. (1999))

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if Token is two digit

    HunTag:
        Type: Token
        Field: Token
        Example: '33' -> [1], '333' -> [0]
        Use case: NER
    """
    return [int(len(form) == 2 and form.isdigit())]


def threeDigitNumOperator(form):
    """Token is three digit

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if Token is three digit

    HunTag:
        Type: Token
        Field: Token
        Example: '333' -> [1], '3333' -> [0]
        Use case: NER
    """
    return [int(len(form) == 3 and form.isdigit())]


def fourDigitNumOperator(form):
    """Token is four digit

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if Token is four digit

    HunTag:
        Type: Token
        Field: Token
        Example: '2015' -> [1], '333' -> [0]
        Use case: NER
    """
    return [int(len(form) == 4 and form.isdigit())]


def isPunctuationOperator(form):
    """Token is punctuation

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if Token is punctuation

    HunTag:
        Type: Token
        Field: Token
        Example: '.' -> [1], 'A.' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int(set(form).issubset(set(',.!"\'():?<>[];{}')))]


def containsDigitAndDashOperator(form):
    """Token contains digit and hyphen (-) (from Bikel et al. (1999))

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if Token contains digits and hyphen (-)

    HunTag:
        Type: Token
        Field: Token
        Example: '2014-15' -> [1], '3-gram' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int(bool(re.match('[0-9]+-[0-9]+', form)))]


def containsDigitAndSlashOperator(form):
    """Token contains digit and slash (/) (from Bikel et al. (1999))

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if Token contains digit and slash (/)

    HunTag:
        Type: Token
        Field: Token
        Example: '2014/2015' -> [1], '3/A' -> [0]
        Use case: NER, but not in SzegedNER
    """
    return [int(bool(re.match('[0-9]+/[0-9]+', form)))]


def containsDigitAndCommaOperator(form):
    """Token contains digit and comma (. or ,) (from Bikel et al. (1999))

    Args:
       form (str): The token

    Returns:
       [Bool in int format]: True if Token contains digit and comma (. or ,)

    HunTag:
        Type: Token
        Field: Token
        Example: '2015.04.07.' -> [1], '2015.txt' -> [0]
        Use case: NER
    """
    return [int(bool(re.match('[0-9]+[,.][0-9]+', form)))]


# XXX lowercased function name: YearDecadeOperator
def yearDecadeOperator(form):
    """Token contains year decade (from Zhou and Su (2002))

    Args:
       form (str): The token

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


def newSentenceStart(sen, _):
    """Sentence starting token or not

    Args:
       sen (list): List of tokens in the sentence

    Returns:
       [[Bool in int format]]: True if Token is at the sentence start

    HunTag:
        Type: Sentence
        Field: Token
        Example: ???
        Use case: NER, Chunk
    """
    featVec = [[0] for _ in sen]
    featVec[0][0] = 1
    return featVec


def newSentenceEnd(sen, _):
    """Sentence ending token or not

    Args:
       sen (list): List of tokens in the sentence

    Returns:
       [[Bool in int format]]: True if Token is at the sentence end

    HunTag:
        Type: Sentence
        Field: Token
        Example: ???
        Use case: NER, Chunk
    """
    featVec = [[0] for _ in sen]
    featVec[-1][0] = 1
    return featVec


### WILL BE DELETED
def OOV(lemma):
    return [int('OOV' in lemma)]


def unknown(anal):
    """Guessed Unknown word from HunMorph?

    Args:
       anal (str): The analysis

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
def getKrLemma(krLemma):
    """Get lemma from HunMorph's analysis

    Args:
       krLemma (str): The token

    Returns:
       [Str]: Returns lemma

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: NER
    """
    return [krLemma.split('/')[0]]


# XXX Return is not bool
def getKrPos(krAnal):
    """Get KR code POS tag

    Args:
       krAnal (str): The token

    Returns:
       [Str]: Returns POS part of KR code

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: NER, but not in SzegedNER
    """
    if '<' in krAnal:
        return [krAnal.split('<')[0]]
    return [krAnal]


# XXX Return is not bool
def getPennTags(pennTag):
    """Reduces Penn tagset's similar tags

    Args:
       pennTag (str): Penn Tag

    Returns:
       [Str]: ???

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: NER
    """
    if re.match('^N', pennTag) or re.match('^PRP', pennTag):
        return ['noun']
    elif pennTag == 'IN' or pennTag == 'TO' or pennTag == 'RP':
        return ['prep']
    elif re.match('DT$', pennTag):
        return ['det']
    elif re.match('^VB', pennTag) or pennTag == 'MD':
        return ['verb']
    else:
        return ['0']


### WILL BE DELETED
def plural(tag):
    return [int(tag == 'NNS' or tag == 'NNPS')]


def HumorPlural(humorTag):
    """Check if Humor code plural

    Args:
       humorTag (str): Humor analysis

    Returns:
       [[Bool in int format]]: True if Humor analysis is plural

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: NER
    """
    return [int('PL' in humorTag)]


### WILL BE DELETED
def getBNCtag(tag):
    return [tag]


# XXX Return is not bool
def getKrEnd(krAnal, _=None):
    """Return KR code end

    Args:
       krAnal (str): The token

    Returns:
       [Str]: ???

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: NER, but not in SzegedNER
    """
    end = '0'
    if '<' in krAnal:
        pieces = krAnal.split('<', 1)
        end = pieces[1]

    return [end]


### WILL BE DELETED
# XXX Return is not bool
# XXX Same as krPatts except since_pos is missing
def sentence_parsePatts(sen, fields, options):
    """parse KR code patterns

    Args:
       sen (list): List of tokens in the sentence
       fields (list): number of fields used
       options (dict): available options

    Returns:
       [[str]]: ???

    HunTag:
        Type: Sentence
        Field: analysis
        Example: ???
        Use case: NER, but not in SzegedNER
    """
    assert len(fields) == 1
    f = fields[0]
    minLength = options['minLength']
    maxLength = options['maxLength']
    rad = options['rad']

    krVec = [tok[f] for tok in sen]
    featVec = [[] for _ in krVec]
    krVecLen = len(krVec)
    # For every token in sentence
    for c in range(krVecLen):
        # Begining in -rad and rad but starts in the list boundaries (lower)
        for k in range(max(-rad, -c), rad):
            # Ending in -rad + 1 and rad + 2  but starts in the list boundaries (upper)
            # and keep minimal and maximal length
            for j in range(max(-rad + 1, minLength + k), min(rad + 2, maxLength + k + 1, krVecLen - c + 1)):
                    value = '+'.join(krVec[c + k:c + j])
                    feat = '{0}_{1}_{2}'.format(k, j, value)
                    featVec[c].append(feat)
    return featVec


def token_pennPlural(pennTag, _=None):
    """Check if Penn code plural

    Args:
       pennTag (str): Penn tag

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
    return [int(pennTag == 'NNS' or pennTag == 'NNPS')]


# XXX Return is not bool
def humorPieces(humorTag):
    """Return Humor tag pieces
    For Humor code:
    'FN/N|PROP|FIRST/ffinev;veznev' => ['FN', 'N|PROP|FIRST', 'ffinev;veznev']
    'FN+PSe3+DEL' => ['FN', 'PSe3', 'DEL']

    Args:
       humorTag (str): Humor tag

    Returns:
       [str]: ???

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: Chunk, NER
    """
    return [item for part in humorTag.split('/') for item in part.split('+')]


# XXX Return is not bool
def humorSimple(humorTag):
    """Humor test: return self

    Args:
       humorTag (str): Humor tag

    Returns:
       [str]: self

    HunTag:
        Type: Token
        Field: analysis
        Example: ???
        Use case: Chunk, NER
    """
    return [humorTag]


# XXX Return is not bool
def wordNetSimple(wordNetTags):
    """Wordnet synsets as tags

    Args:
       wordNetTags (str): wordNet synsets as tags

    Returns:
       [str]: list of wordNet synsets as features

    HunTag:
        Type: Token
        Field: WordNet tags
        Example: animate.n.1/human.n.1 -> ['animate.n.1', 'human.n.1'], '' -> []
        Use case: Chunk, NER
    """
    if len(wordNetTags) == 0:
        return []
    return wordNetTags.split('/')

MMOpatt = re.compile('[\[,\]]')


# XXX Return is not bool
def mmoSimple(mmoTags):
    """MMO properties

    Args:
       mmoTags (str): wordNet synsets as tags

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
    if mmoTags == '-':
        return []
    return [x for x in MMOpatt.split(mmoTags) if x]


### WILL BE DELETED
# XXX This is never used. Will be deleted
# Slow, because the combinatoric explosion...
def myPatts(sen, fields, options, fullKr=False):
    lang = options['lang']
    assert lang in ('en', 'hu')
    minLength = int(options['minLength'])
    maxLength = int(options['maxLength'])
    rad = int(options['rad'])

    # print 'sen:'
    # print sen
    # print 'fields:'
    # print fields
    # print 'fullKr:'
    # print fullKr

    # assert len(fields) == 1
    # f = fields[0]
    featVec = [[] for _ in sen]
    # sys.stderr.write(str(len(sen))+'words\n')
    for c in range(len(sen)):
        # Múlt    múlt    FN      O       B-N_2+
        # print('vektor: ')
        # print(krVec[c])
        # spec = krVec[ckrVec[c]].split('#')
        # print(spec)
        # if (krVec[c] != 'O'):
        #     featVec[c].append(krVec[c])
        #     continue
        # Begining in -rad and rad but starts in the list boundaries (lower)
        for k in range(max(-rad, -c), rad):
            # Ending in -rad + 1 and rad + 2  but starts in the list boundaries (upper)
            # and keep minimal and maximal length
            for j in range(max(-rad + 1, minLength + k), min(rad + 2, maxLength + k + 1, len(sen) - c + 1)):
                # sys.stderr.write('*')
                seqs = []
                for curr in range(c + k, c + j):
                    seq = []
                    for f in fields:
                        seq.append(sen[curr][f])
                    # Every elem is appended to 'seq', in every possible combination...
                    if len(seqs) == 0:  # seq2 = seq
                        seqs = seq
                    else:
                        ujelemek = []
                        for v in seqs:  # already made sequences
                            # print('elems already in: ' + str(v) + 'seq: ' + str(seq))
                            for elem in seq:  # There is a new element
                                eddigi = deepcopy(v)
                                if isinstance(eddigi, list):
                                    eddigi.append(elem)
                                else:
                                    eddigi = [eddigi, elem]  # new elem = [old elem]
                                # ujelem.append(elem)
                                ujelemek.append(eddigi)
                        # print('most ' + str(seqs) + str(ujelemek))
                        seqs = ujelemek
                        # print('most ' + str(seqs))
                    # print(seqs) #
                    for u in seqs:
                        value = '+'.join(u)
                        feat = '{0}_{1}_{2}'.format(k, j, value)
                        featVec[c].append(feat)
                # sys.stderr.write('\n')
    # print 'featVec:'
    # print featVec
    return featVec


# XXX never used
# XXX Return is not bool
# XXX and not list
def tags_since_pos(sen, tokRange, myPos='DT', strict=True):
    """Sentence ending token or not

    Args:
       sen (list): List of tokens in the sentence
       tokRange (int): range of tokens from the start of the sentence
       myPos (str):
       strict(bool): ...

    Returns:
       [[Bool in int format]]: True if Token is at the sentence end

    HunTag:
        Type: Sentence
        Field: analysis
        Example: ???
        Use case: NER, Chunk

    Replaces:
        since_dt: abstraction
    """
    tags = []
    for pos in sen[:tokRange]:
        if (strict and myPos == pos) or (not strict and re.search(myPos, pos)):
            tags = [pos]
        else:
            tags.append(pos)
    return '+'.join(tags)


birtokos = re.compile(r'--[sp]\d')


### WILL BE DELETED
# Mypatts tesója
# Krpatts kis módosítással
# Két szó egymás mellett más esetben van.
# Birtokos és a birtok kereső is
def mySpecPatts(sen, fields, options, fullKr=False):
    # XXX Tests, to be included in comment
    # s = [['egy', '[Tf]', 'egyf2'], ['ketto', 'N--s2kettof1', 'kettof2'], ['harom', 'N--p3haromf1', 'haromf3'],
    #  ['negy', 'negyf1', 'negyf2']]
    # o = {'lang':'hu', 'minLength': 2, 'maxLength' : 99, 'rad' : 2}
    # print(krPatts(s, [1], o))
    # print(mySpecPatts(s, [1], o))
    # print(myPatts(s, [1,2], o))
    assert options['lang'] in ('en', 'hu')
    minLength = int(options['minLength'])
    maxLength = int(options['maxLength'])
    rad = int(options['rad'])
    assert len(fields) == 1
    f = fields[0]
    featVec = [[] for _ in sen]
    # XXX TODO: make it a parameter
    # Use full KR code. It yielded better results.
    krVec = [tok[f] for tok in sen]
    """
    if lang == 'hu':
        if not fullKr:
            krVec = [getPosTag(kr) for kr in krVec]
    else:
        krVec = [tok[f][0] for tok in sen]
    """

    assert options['lang'] in ('en', 'hu')
    minLength = int(options['minLength'])
    maxLength = int(options['maxLength'])
    rad = int(options['rad'])
    assert len(fields) == 1
    f = fields[0]
    featVec = [[] for _ in sen]
    # XXX TODO: make it a parameter
    # Use full KR code. It yielded better results.
    krVec = [tok[f] for tok in sen]
    """
    if lang == 'hu':
        if not fullKr:
            krVec = [getPosTag(kr) for kr in krVec]
    else:
        krVec = [tok[f][0] for tok in sen]
    """

    assert len(krVec) == len(sen)
    krVecLen = len(krVec)
    # For every token in sentence
    for c in range(krVecLen):
        # since_dt using since_dt

        tagst = tags_since_pos(krVec, c, '[Tf]')  # [Tf], DT
        if len(tagst) > 0:
            featVec[c].append('dt_' + tagst)

        # XXX EZ A PÁR SOR KÜLÖNBÖZIK A KRpatts-tól
        lastF = '' if c == 0 else krVec[c - 1]
        if lastF.startswith('[N') and krVec[c].startswith('[N') and lastF != krVec[c]:
            featVec[c].append('FNelter')  # XXX LEFORDÍTANI
        if birtokos.search(krVec[c]):
            tagst = tags_since_pos(krVec, c, '^\[?N', False)
            if len(tagst) > 0:
                featVec[c].append('birtok_' + tagst)

        # Begining in -rad and rad but starts in the list boundaries (lower)
        for k in range(max(-rad, -c), rad):
            # Ending in -rad + 1 and rad + 2  but starts in the list boundaries (upper)
            # and keep minimal and maximal length
            for j in range(max(-rad + 1, minLength + k), min(rad + 2, maxLength + k + 1, krVecLen - c + 1)):
                value = '+'.join(krVec[c + k:c + j])
                feat = '{0}_{1}_{2}'.format(k, j, value)
                featVec[c].append(feat)
    return featVec
