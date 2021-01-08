#!/bin/bash -o pipefail

RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
NOCOLOR=$(tput sgr0)

VENVPYTHON=$1
MODULE=$2
CURDIR=$3

echo "Running train tests..."
# train
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} train --model=testMNP --config-file=configs/maxnp.szeged.emmorph.yaml \
    --gold-tag-field gold -i ${CURDIR}/tests/test.maxnp.emmorph 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} train --model=testNER --config-file=configs/ner.szeged.emmorph.yaml \
    --gold-tag-field gold -i ${CURDIR}/tests/test.ner.emmorph 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
# train, featurize (for crfsuite)
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} train-featurize --model=testMNP \
    --config-file=configs/maxnp.szeged.emmorph.yaml \
    --gold-tag-field gold -i ${CURDIR}/tests/test.maxnp.emmorph | \
    diff -sy --suppress-common-lines - ${CURDIR}/tests/test.maxnp.CRFsuite.train 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} train-featurize --model=testNER \
    --config-file=configs/ner.szeged.emmorph.yaml \
    --gold-tag-field gold -i ${CURDIR}/tests/test.ner.emmorph | \
    diff -sy --suppress-common-lines - ${CURDIR}/tests/test.ner.CRFsuite.train 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
# most-informative-features
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} most-informative-features --model=testMNP \
    --config-file=configs/maxnp.szeged.emmorph.yaml -i ${CURDIR}/tests/test.maxnp.emmorph | \
    diff -sy --suppress-common-lines - ${CURDIR}/tests/test.maxnp.mostInformativeFeatures 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} most-informative-features --model=testNER \
    --config-file=configs/ner.szeged.emmorph.yaml -i ${CURDIR}/tests/test.ner.emmorph | \
    diff -sy --suppress-common-lines - ${CURDIR}/tests/test.ner.mostInformativeFeatures 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
# transmodel-train
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} transmodel-train --model=testMNP \
    --config-file=configs/maxnp.szeged.emmorph.yaml \
    --gold-tag-field gold -i ${CURDIR}/tests/test.maxnp.emmorph 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
    # --trans-model-order [2 or 3, default: 3]
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} transmodel-train --model=testNER \
    --config-file=configs/ner.szeged.emmorph.yaml \
    --gold-tag-field gold -i ${CURDIR}/tests/test.ner.emmorph 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
    # --trans-model-order [2 or 3, default: 3]

echo "Running eval tests..."
# tag
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} tag --model=models/maxnp.szeged.emmorph \
    --config-file=configs/maxnp.szeged.emmorph.yaml --label-tag-field NP-BIO \
    -i ${CURDIR}/tests/test.maxnp.emmorph | \
    diff -sy --suppress-common-lines - ${CURDIR}/tests/test.maxnp.tag 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} tag --model=models/ner.szeged.emmorph \
    --config-file=configs/ner.szeged.emmorph.yaml \
    --label-tag-field NER-BIO -i ${CURDIR}/tests/test.ner.emmorph | \
    diff -sy --suppress-common-lines - ${CURDIR}/tests/test.ner.tag 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
# tag, featurize (for crfsuite)
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} tag-featurize --model=models/maxnp.szeged.emmorph \
    --config-file=configs/maxnp.szeged.emmorph.yaml -i ${CURDIR}/tests/test.maxnp.emmorph | \
    diff -sy --suppress-common-lines - ${CURDIR}/tests/test.maxnp.CRFsuite.tag 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} tag-featurize --model=models/ner.szeged.emmorph \
    --config-file=configs/ner.szeged.emmorph.yaml -i ${CURDIR}/tests/test.ner.emmorph | \
    diff -sy --suppress-common-lines - ${CURDIR}/tests/test.ner.CRFsuite.tag 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
# tag FeatureWeights
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} print-weights -w 100 --model=models/maxnp.szeged.emmorph \
    --config-file=configs/maxnp.szeged.emmorph.yaml | \
    diff -sy --suppress-common-lines - ${CURDIR}/tests/test.maxnp.modelWeights 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
time (cd /tmp && ${VENVPYTHON} -m ${MODULE} print-weights -w 100 --model=models/ner.szeged.emmorph \
    --config-file=configs/ner.szeged.emmorph.yaml | \
    diff -sy --suppress-common-lines - ${CURDIR}/tests/test.ner.modelWeights 2>&1 | head -n100) \
    && echo "${GREEN}Test OK${NOCOLOR}" || exit 1
