# Bash is needed for time
SHELL := /bin/bash -o pipefail
DIR := ${CURDIR}
red := $(shell tput setaf 1)
green := $(shell tput setaf 2)
sgr0 := $(shell tput sgr0)
all:
	@echo "See Makefile for possible targets!"

dist/*.whl dist/*.tar.gz:
	@echo "Building package..."
	python3 setup.py sdist bdist_wheel

build: dist/*.whl dist/*.tar.gz

install-user: build
	@echo "Installing package to user..."
	pip3 install dist/*.whl

test-train:
	@echo "Running train tests..."
	# train
	cd /tmp && python3 -m huntag train --model=testMNP --config-file=configs/maxnp.szeged.emmorph.yaml \
						--gold-tag-field gold -i $(DIR)/tests/test.maxnp.emmorph
	@echo "$(green)Test OK$(sgr0)"
	cd /tmp && python3 -m huntag train --model=testNER --config-file=configs/ner.szeged.emmorph.yaml \
						--gold-tag-field gold -i $(DIR)/tests/test.ner.emmorph
	@echo "$(green)Test OK$(sgr0)"
	# train, featurize (for crfsuite)
	cd /tmp && python3 -m huntag train-featurize --model=testMNP --config-file=configs/maxnp.szeged.emmorph.yaml \
						--gold-tag-field gold -i $(DIR)/tests/test.maxnp.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.maxnp.CRFsuite.train 2>&1 | head -n100
	@echo "$(green)Test OK$(sgr0)"
	cd /tmp && python3 -m huntag train-featurize --model=testNER --config-file=configs/ner.szeged.emmorph.yaml \
						--gold-tag-field gold -i $(DIR)/tests/test.ner.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.ner.CRFsuite.train 2>&1 | head -n100
	@echo "$(green)Test OK$(sgr0)"
	# most-informative-features
	cd /tmp && python3 -m huntag most-informative-features --model=testMNP \
						--config-file=configs/maxnp.szeged.emmorph.yaml -i $(DIR)/tests/test.maxnp.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.maxnp.mostInformativeFeatures 2>&1 | head -n100
	@echo "$(green)Test OK$(sgr0)"
	cd /tmp && python3 -m huntag most-informative-features --model=testNER \
						--config-file=configs/ner.szeged.emmorph.yaml -i $(DIR)/tests/test.ner.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.ner.mostInformativeFeatures 2>&1 | head -n100
	@echo "$(green)Test OK$(sgr0)"
	# transmodel-train
	cd /tmp && python3 -m huntag transmodel-train --model=testMNP --config-file=configs/maxnp.szeged.emmorph.yaml \
						--gold-tag-field gold -i $(DIR)/tests/test.maxnp.emmorph
						# --trans-model-order [2 or 3, default: 3]
	@echo "$(green)Test OK$(sgr0)"
	cd /tmp && python3 -m huntag transmodel-train --model=testNER --config-file=configs/ner.szeged.emmorph.yaml \
						--gold-tag-field gold -i $(DIR)/tests/test.ner.emmorph
						# --trans-model-order [2 or 3, default: 3]
	@echo "$(green)Test OK$(sgr0)"

test-eval:
	@echo "Running eval tests..."
	# tag
	cd /tmp && python3 -m huntag tag --model=models/maxnp.szeged.emmorph \
						--config-file=configs/maxnp.szeged.emmorph.yaml --label-tag-field NP-BIO \
						-i $(DIR)/tests/test.maxnp.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.maxnp.tag 2>&1 | head -n100
	@echo "$(green)Test OK$(sgr0)"
	cd /tmp && python3 -m huntag tag --model=models/ner.szeged.emmorph --config-file=configs/ner.szeged.emmorph.yaml \
						--label-tag-field NER-BIO -i $(DIR)/tests/test.ner.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.ner.tag 2>&1 | head -n100
	@echo "$(green)Test OK$(sgr0)"
	# tag, featurize (for crfsuite)
	cd /tmp && python3 -m huntag tag-featurize --model=models/maxnp.szeged.emmorph \
						--config-file=configs/maxnp.szeged.emmorph.yaml -i $(DIR)/tests/test.maxnp.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.maxnp.CRFsuite.tag 2>&1 | head -n100
	@echo "$(green)Test OK$(sgr0)"
	cd /tmp && python3 -m huntag tag-featurize --model=models/ner.szeged.emmorph \
						--config-file=configs/ner.szeged.emmorph.yaml -i $(DIR)/tests/test.ner.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.ner.CRFsuite.tag 2>&1 | head -n100
	@echo "$(green)Test OK$(sgr0)"
	# tag FeatureWeights
	cd /tmp && python3 -m huntag print-weights -w 100 --model=models/maxnp.szeged.emmorph \
						--config-file=configs/maxnp.szeged.emmorph.yaml | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.maxnp.modelWeights 2>&1 | head -n100
	@echo "$(green)Test OK$(sgr0)"
	cd /tmp && python3 -m huntag print-weights -w 100 --model=models/ner.szeged.emmorph \
						--config-file=configs/ner.szeged.emmorph.yaml | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.ner.modelWeights 2>&1 | head -n100
	@echo "$(green)Test OK$(sgr0)"

test: test-train test-eval

install-user-test: install-user test
	@echo "The test was completed successfully!"

ci-test: install-user-test

uninstall:
	@echo "Uninstalling..."
	pip3 uninstall -y huntag

install-user-test-uninstall: install-user-test uninstall

clean:
	rm -rf dist/ build/ huntag.egg-info/

clean-build: clean build
