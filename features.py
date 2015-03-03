#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-
"""
features.py stores implementations of inidvidual feature types for use with
HunTag. Feel free to add your own features but please add comments to describe
them.
"""

import re
import sys

nonalf = re.compile(r'\W+')
casCode = re.compile(r'CAS<...>')


def stupidStem(form):
    r = form.rfind("-")
    if r == -1:
        return form
    else:
        return form[:r]


# The original isCapitalizedOperator, but this checks only
# if it has uppercase in it anywhere
def isCapitalizedOperator(form):
    return [int(form.lower() != form)]


# I gave it a new name
def hasCapOperator(form):
    return [int(form.lower() != form)]


# New: all lowercase
def lowerCaseOperator(form):
    return [int(form.lower() == form)]


# This is the new isCapitalized: Starts with uppercase
def isCapOperator(form):
    return [int(form[0] != form[0].lower())]


# This is new: Starts lowerase
def notCapitalizedOperator(form):
    return [int(form[0] == form[0].lower())]


def isAllcapsOperator(form):
    return [int(stupidStem(form).isupper())]


# The first letter is lower, the others has, but not all uppercase...
def isCamelOperator(form):
    return [int(form[1].islower() and not form[1:].isupper() and not form[1:].islower())]


def threeCaps(form):
    return [int(len(form) == 3 and stupidStem(form).isupper())]


def startsWithNumberOperator(form):
    return [int(form[0].isdigit())]


def isNumberOperator(form):
    return [int(set(stupidStem(form)).issubset(set('0123456789,.-%')))]


def hasNumberOperator(form):
    return [int(not set('0123456789').isdisjoint(set(form)))]


def hasDashOperator(form):
    return [int('-' in set(form))]


def hasUnderscoreOperator(form):
    return [int('_' in set(form))]


def hasPeriodOperator(form):
    return [int('.' in set(form))]


def sentenceStart(surfaceformVector):
    featureVector = []
    for formIndex, form in enumerate(surfaceformVector):
        if form == '':
            featureVector.append('')
        else:
            if formIndex > 0:
                val = int(surfaceformVector[formIndex - 1] == '')
            else:
                val = 1
            featureVector.append(val)
    return featureVector


def sentenceEnd(surfaceformVector):
    featureVector = []
    for formIndex, form in enumerate(surfaceformVector):
        if form == '':
            featureVector.append('')
        else:
            if formIndex + 1 < len(surfaceformVector):
                val = int(surfaceformVector[formIndex + 1] == '')
            else:
                val = 1
            featureVector.append(val)
    return featureVector

# GLOBAL DECLARATION BEGIN
smallcase = 'aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz'
bigcase = 'AÁBCDEÉFGHIÍJKLMNOÓÖŐPQRSTUÚÜŰVWXYZ'
big2small = {}
for i, _ in enumerate(bigcase):
    big2small[bigcase[i]] = smallcase[i]
# GLOBAL DECLARATION END


def longPattern(form):
    pattern = ''
    for char in form:
        if char in smallcase:
            pattern += 'a'
        elif char in bigcase:
            pattern += 'A'
        else:
            pattern += '_'

    return [pattern]


def shortPattern(form):
    pattern = ''
    prev = ''
    for char in form:
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


def scase(word):
    """Make range lowercase"""
    for j in range(35):
        if word[0] == bigcase[j]:
            return smallcase[j] + word[1:]

    return [word]


def isInRangeWithSmallCase(word):
    # print(word, file=sys.stderr, flush=True)
    checkedRange = 30
    if word[0] in smallcase:
        return 'n/a'
    else:
        return [int(db.isInRange(scase(word), checkedRange, wordcount))]


def chunkTag(chunktag):
    return [chunktag]


def chunkType(chunktag):
    return [chunktag[2:]]


def chunkPart(chunktag):
    return [chunktag[0]]


def getForm(word):
    if '_' not in word:
        return [word]
    return ['MERGED']


def ngrams(word, options):
    n = int(options['n'])
    f = []
    for c in range(max(0, len(word) - n + 1)):
        if c == 0:
            f.append('@{0}'.format(str(word[c:c + n])))
        elif c + n == len(word):
            f.append('{0}@'.format(str(word[c:c + n])))
        else:
            f.append(str(word[c:c + n]))
    return f


def firstChar(word):
    return [word[0]]


def msdPos(msd):
    return msd[1]


def msdPosAndChar(msd):
    pos = msd[1]
    f = []
    for c, ch in enumerate(msd[2:-1]):
        if ch == '-':
            pass
        else:
            f.append('{0}{1}{2}'.format(pos, str(c), ch))
    return f


def prefix(word, options):
    n = int(options['n'])
    return [word[0:n]]


def suffix(word, options):
    n = int(options['n'])
    return [word[-n:]]


def lemmaLowered(word, lemma):
    if word[0] not in bigcase:
        if lemma[0] in bigcase and big2small[lemma[0]] == word[0]:
            return ['raised']
        return ['N/A']

    if word[0] == lemma[0]:
        return [0]

    if big2small[word[0]] == lemma[0]:
        return [1]

    return ['N/A']


def krPieces(kr):
    pieces = nonalf.split(kr.split('/')[-1])
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


def fullKrPieces(kr):
    return krPieces('/'.join(kr.split('/')[1:]))


def krFeats(kr):
    pieces = nonalf.split(kr)[1:]
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


def krConjs(kr):
    pieces = nonalf.split(kr)
    # feats = []
    conjs = []

    for ind, e1 in enumerate(pieces):
        for e2 in pieces[ind + 1:]:
            if e2 == '':
                continue
            conjs.append('{0}+{1}'.format(e1, e2))

    return [feat for feat in conjs if feat]


def capsPattern(sentence, fields):
    featVec = [[] for _ in sentence]

    assert len(fields) == 1
    tokens = [word[fields[0]] for word in sentence]
    upperFlags = [isCapitalizedOperator(token)[0] for token in tokens]
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


def capsPattern_test():
    tokens = 'A certain Ratio Of GDP is Gone Forever'.split()
    sentence = [[token] for token in tokens]
    fields = [0]
    print(capsPattern(sentence, fields))


def isBetweenSameCases(sentence, fields, maxDist=6):
    featVec = [[] for _ in sentence]
    if len(fields) > 1:
        print('Error: "isBetweenSameCases" function\'s "fields" argument\'s\
            length must be one not {0}'.format(len(fields)), file=sys.stderr,
              flush=True)
        sys.exit(1)

    krVec = [token[fields[0]] for token in sentence]
    nounCases = [[] for _ in sentence]

    # print krVec

    for c, kr in enumerate(krVec):
        if 'CAS' not in kr:
            # if 'NOUN' not in kr:
            continue
        cases = casCode.findall(kr)
        if not cases:
            nounCases[c] = ['NO_CASE']
        else:
            case = cases[0][-4:-1]
            nounCases[c] = [case]

    # print nounCases

    leftCase = {}
    rightCase = {}
    currCase = None
    casePos = None
    for j, _ in enumerate(sentence):
        if not nounCases[j]:
            leftCase[j] = (currCase, casePos)
        else:
            currCase = nounCases[j]
            casePos = j
            leftCase[j] = (None, None)

    # print leftCase

    currCase = None
    casePos = None
    for j in range(len(sentence) - 1, -1, -1):
        if not nounCases[j]:
            rightCase[j] = (currCase, casePos)
        else:
            currCase = nounCases[j]
            rightCase[j] = (None, None)
            casePos = j

    # print rightCase

    for j, _ in enumerate(sentence):
        if rightCase[j][0] == leftCase[j][0] and rightCase[j][0] is not None:
            if abs(rightCase[j][1] - leftCase[j][1]) <= maxDist:
                featVec[j] = [1]
                continue
        featVec[j] = [0]

    return featVec


def getPosTag(kr):
    if '/' in kr:
        return getPosTag(kr.split('/')[-1])
    else:
        return nonalf.split(kr)[0]


def krPatts(sen, fields, options, fullKr=False):
    lang = options['lang']
    assert lang in ('en', 'hu')
    minLength = int(options['minLength'])
    maxLength = int(options['maxLength'])
    rad = int(options['rad'])

    assert len(fields) == 1
    f = fields[0]
    featVec = [[] for _ in sen]
    krVec = [tok[f] for tok in sen]
    if lang == 'hu':
        if not fullKr:
            krVec = [getPosTag(kr) for kr in krVec]
    else:
        krVec = [tok[f][0] for tok in sen]

    assert len(krVec) == len(sen)
    # print(str(len(sen))+'words', file=sys.stderr, flush=True)
    krVecLen = len(krVec)
    for c in range(krVecLen):
        # print('@')
        # print('word '+str(c), file=sys.stderr, flush=True)

        # XXX SHOULD LIMIT RADIUS TO THE BOUNDS!!!!
        for k in range(-rad, rad):
            for j in range(-rad + 1, rad + 2):
                # print(str(i)+' '+str(j), end='', file=sys.stderr, flush=True)
                a = c + k
                b = c + j

                # if b-a == 3:
                # print(str(c)+'\t'+str(i)+' '+str(j), file=sys.stderr, flush=True)

                if a >= 0 and b <= krVecLen and minLength <= b - a <= maxLength:
                    # print('*', end='', file=sys.stderr, flush=True)
                    value = '+'.join([krVec[x] for x in range(a, b)])
                    feat = '_'.join((str(k), str(j), value))
                    featVec[c].append(feat)
                    # print('', file=sys.stderr, flush=True)
    return featVec


def getTagType(tag):
    return tag[2:]


def posStart(postag):
    return postag[0]


def posEnd(postag):
    return postag[-1]


def getNpPart(partchunkTag):
    if partchunkTag == 'O' or partchunkTag[2:] != 'NP':
        return 'O'
    else:
        return partchunkTag[0]


# from Bikel et al. (1999)
def CapPeriodOperator(form):
    return [int(bool(re.match(r'[A-Z]\.$', form)))]


def isDigitOperator(form):
    return [int(form.isdigit())]


# from Zhou and Su (2002)
def oneDigitNumOperator(form):
    return [int(len(form) == 1 and form.isdigit())]


# from Bikel et al. (1999)
def twoDigitNumOperator(form):
    return [int(len(form) == 2 and form.isdigit())]


def threeDigitNumOperator(form):
    return [int(len(form) == 3 and form.isdigit())]


def fourDigitNumOperator(form):
    return [int(len(form) == 4 and form.isdigit())]


def isPunctuationOperator(form):
    return [int(set(form).issubset(set(',.!"\'():?<>[];{}')))]


# from Bikel et al. (1999)
def containsDigitAndDashOperator(form):
    return [int(re.match('[0-9]+-[0-9]+', form))]


# from Bikel et al. (1999)
def containsDigitAndSlashOperator(form):
    return [int(re.match('[0-9]+/[0-9]+', form))]


# from Bikel et al. (1999)
def containsDigitAndCommaOperator(form):
    return [int(re.match('[0-9]+[,.][0-9]+', form))]


# from Zhou and Su (2002)
def YearDecadeOperator(form):
    return [int(re.match('[0-9][0-9]s$', form) or
                re.match('[0-9][0-9][0-9][0-9]s$', form))]


# XXX I wonder the original author wanted...
def newSentenceStart(sen, _):
    featVec = []
    for tok in sen:
        if tok is sen[0]:
            featVec.append([1])
        else:
            featVec.append([0])
    return featVec


# XXX I wonder the original author wanted...
def newSentenceEnd(sen, _):
    featVec = []
    for tok in sen:
        if tok is sen[-1]:
            featVec.append([1])
        else:
            featVec.append([0])
        return featVec


def OOV(lemma):
    return [int('OOV' in lemma)]


def getKrLemma(lemma):
    return [lemma.split('/')[0]]


def getKrPos(kr):
    if '<' in kr:
        pieces = kr.split('<')
        pos = pieces[0]
    else:
        pos = kr
    return [pos]


def getPennTags(tag):
    if re.match('^N', tag) or re.match('^PRP', tag):
        return ['noun']
    elif tag == 'IN' or tag == 'TO' or tag == 'RP':
        return ['prep']
    elif re.match('DT$', tag):
        return ['det']
    elif re.match('^VB', tag) or tag == 'MD':
        return ['verb']
    else:
        return ['0']


def plural(tag):
    return [int(tag == 'NNS' or tag == 'NNPS')]


def getBNCtag(tag):
    return [tag]

