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
    if isAllcapsOperator(form):
        return [int(False)]
    else:
        return [int(form[1:].lower() != form[1:])]


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


def isBetweenSameCases(sentence, fields, params={'maxDist': '6'}):
    maxDist = int(params['maxDist'])
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
        if cases == []:
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
        if nounCases[j] == []:
            leftCase[j] = (currCase, casePos)
        else:
            currCase = nounCases[j]
            casePos = j
            leftCase[j] = (None, None)

    # print leftCase

    currCase = None
    casePos = None
    for j in range(len(sentence) - 1, -1, -1):
        if nounCases[j] == []:
            rightCase[j] = (currCase, casePos)
        else:
            currCase = nounCases[j]
            rightCase[j] = (None, None)
            casePos = j

    # print rightCase

    for j, _ in enumerate(sentence):
        featVec[j] = [0]
        if (rightCase[j][0] == leftCase[j][0]) and (rightCase[j][0] is not None):
            if abs(rightCase[j][1] - leftCase[j][1]) <= maxDist:
                featVec[j] = [1]

    return featVec


def getPosTag(kr):
    if '/' in kr:
        return getPosTag(kr.split('/')[-1])
    else:
        return nonalf.split(kr)[0]

def tags_since_dt(sentence, i):
    """
	Last determinant feature:
    All the POS tags that occured since the last determinant is joined with '+'
	Determinant: XXX TODO: Make this a parameter
    English: 'DT'
    Hungarian (KR code): '[Tf]'
    """
    tags = set()
    for pos in sentence[:i]:
        if pos == 'DT':# [Tf], DT
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))

def krPatts(sen, fields, options, fullKr=False):
    lang = options['lang']
    assert lang in ('en', 'hu')
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
    # print(str(len(sen))+'words', file=sys.stderr, flush=True)
    krVecLen = len(krVec)
    for c in range(krVecLen):
        # since_dt using since_dt
        tagst = tags_since_dt(krVec, c)
        if (len(tagst) > 0):
            featVec[c].append('dt_' + tagst) 
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


def krPlural(tag):
        return [int('NOUN<PLUR' in tag)]


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
def capPeriodOperator(form):
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
    return [int(bool(re.match('[0-9]+-[0-9]+', form)))]


# from Bikel et al. (1999)
def containsDigitAndSlashOperator(form):
    return [int(bool(re.match('[0-9]+/[0-9]+', form)))]


# from Bikel et al. (1999)
def containsDigitAndCommaOperator(form):
    return [int(bool(re.match('[0-9]+[,.][0-9]+', form)))]


# from Zhou and Su (2002)
def yearDecadeOperator(form):
    return [int(bool(re.match('[0-9][0-9]s$', form) or
                re.match('[0-9][0-9][0-9][0-9]s$', form)))]


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


def unknown(word):
    return [int('UNKNOWN' in word or 'OOV' in word)]


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


def HumorPlural(tag):
    return [int('PL' in tag)]


def getBNCtag(tag):
    return [tag]

# For Humor code:
#    "FN/N|PROP|FIRST/ffinev;veznev" => ["FN", "N|PROP|FIRST", "ffinev;veznev"]
#    "FN+PSe3+DEL" => ["FN", "PSe3", "DEL"]
def humorPieces(c) :
    parts = c.split('/')
    if len(parts) == 1 :
        return c.split('+')

    feats=[]
    for i in parts:
        feats.extend(i.split('+'))
    return feats

# Humor test: return self
def humorSimple(c) :
    return [c]

# WordNet test
def wordNetSimple(c) :
    if (len(c) == 0):
        return []
    return c.split('/')


# mmo properties
# in:'NX[abstract=YES,animate=NIL,auth=YES,company=NIL,encnt=YES,human=NIL]'
# out: ['NX', 'abstract=YES', 'animate=NIL', 'auth=YES', 'company=NIL', 'encnt=YES', 'human=NIL']
def mmoSimple(c):
    if (c=='-'):
        return []
    p=re.compile('[\[,\]]').split(c)
#    p = filter(bool, p)
    return [x for x in p if x]

# XXX This is never used. Will be deleted 
# Slow, because the combinatoric explosion...
def myPatts(sen, fields, options, fullKr=False):
    lang = options['lang']
    assert lang in ('en', 'hu')
    minLength = int(options['minLength'])
    maxLength = int(options['maxLength'])
    rad = int(options['rad'])


#    print 'sen:'
#    print sen
#    print 'fields:'
#    print fields
#    print 'fullKr:'
#    print fullKr

#    assert len(fields)==1
#    f = fields[0]
    featVec = [ [] for tok in sen]
    #sys.stderr.write(str(len(sen))+'words\n')
    for c in range(len(sen)):

#M.lt    m.lt    FN      O       B-N_2+
#        print('vektor: ')
#        print(krVec[c])
#        spec = krVec[ckrVec[c]].split('#')
#        print(spec)
#        if (krVec[c] != 'O'):
#            featVec[c].append(krVec[c])
#            continue
        for i in range (-rad, rad):
            for j in range(-rad+1, rad+2):
        #sys.stderr.write(str(i)+' '+str(j))
                a = c+i
                b = c+j

                if a >= 0 and b <= len(sen) and b-a >= minLength and b-a <= maxLength:
                #sys.stderr.write('*')
                    seqs = []
                    for curr in range(c+i, c+j):
                        seq = []
                        for f in fields:
                            seq.append(sen[curr][f])
                        # Every elem is appended to 'seq', in every possible combination...
                        if len(seqs) == 0: #seq2 = seq
                            seqs = seq
                        else:
                            ujelemek = []
                            for v in seqs: # already made sequences
#                                print('elems already in: ' + str(v) + 'seq: ' + str(seq))
                                for elem in seq: # There is a new element
                                    eddigi = deepcopy(v)
                                    if (isinstance(eddigi, list)):
                                        eddigi.append(elem)
                                    else:
                                        eddigi = [eddigi, elem] # new elem = [old elem]
#                                    ujelem.append(elem)
                                    ujelemek.append(eddigi)
#                            print('most ' + str(seqs) + str(ujelemek))
                            seqs = ujelemek
#                            print('most ' + str(seqs))
                    #print(seqs) #
                    for u in seqs:
                        value = '+'.join(u)
                        feat = '_'.join( (str(i), str(j), value) )
                        featVec[c].append(feat)
                #sys.stderr.write('\n')
#    print 'featVec:'
#    print featVec
    return featVec

# since_dt abstraction
# XXX never used
def tags_since_pos(sentence, i, myPos='DT', strict=1):
    tags = []
    for pos in sentence[:i]:
        if (strict==1 and myPos == pos) or (strict==0 and re.search(myPos, pos)):
            tags = [pos]
        else:
            tags.append(pos)
    return '+'.join(tags)

# Mypatts tesója
# Krpatts kis módosítással
# Két szó egymás mellett más esetben van.
# Birtokos és a birtok kereső is
def mySpecPatts(sen, fields, options, fullKr=False):
    lang = options['lang']
    assert lang in ('en', 'hu')
    minLength = int(options['minLength'])
    maxLength = int(options['maxLength'])
    rad = int(options['rad'])

    assert len(fields) == 1
    f = fields[0]
    featVec = [[] for _ in sen]
    krVec = [tok[f] for tok in sen]
#    if lang == 'hu':
#        if not fullKr:
#            krVec = [getPosTag(kr) for kr in krVec]
#    else:
#        krVec = [tok[f][0] for tok in sen]
    krVec = [tok[f] for tok in sen]

    birtokos = re.compile(r'--[sp]\d')
    assert len(krVec) == len(sen)
    # print(str(len(sen))+'words', file=sys.stderr, flush=True)
    krVecLen = len(krVec)
    for c in range(krVecLen):


        tagst = tags_since_pos(krVec, c, '[Tf]')
        if (len(tagst)>0):
            featVec[c].append('dt_' + tagst)
        # print('@')
        # print('word '+str(c), file=sys.stderr, flush=True)
        #M.lt    m.lt    FN      O       B-N_2+
#        print('vektor: ')
#        print(krVec[c])
#        spec = krVec[c].split('#')
#        print(spec)
#        if (len(spec) > 1 and spec[0] != 'O'):
#            featVec[c].append(spec[0])
#            continue
        lastF = '' if c==0 else krVec[c-1]
        if (lastF.startswith('[N') and krVec[c].startswith('[N') and lastF != krVec[c]):
            featVec[c].append('FNelter')
        if (birtokos.search(krVec[c])):
            tagst = tags_since_pos(krVec, c, '^\[?N', 0)
            if (len(tagst)>0):
                featVec[c].append('birtok_' + tagst)

#            featVec[c].append('birtok') # all tags should be enumerated in between

        # XXX SHOULD LIMIT RADIUS TO THE BOUNDS!!!!
        for k in range(-rad, rad):
            #lastF = krVec[k]
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

# XXX Tests, to be included in comment 
#s = [['egy', '[Tf]', 'egyf2'], ['ketto', 'N--s2kettof1', 'kettof2'], ['harom', 'N--p3haromf1', 'haromf3'], ['negy', 'negyf1', 'negyf2']]
#o = {'lang':'hu', 'minLength': 2, 'maxLength' : 99, 'rad' : 2}
#print(krPatts(s, [1], o))
#print(mySpecPatts(s, [1], o))
#print(myPatts(s, [1,2], o))
