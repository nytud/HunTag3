# Bash is needed for time
SHELL := /bin/bash -o pipefail
DIR := ${CURDIR}
red := $(shell tput setaf 1)
green := $(shell tput setaf 2)
sgr0 := $(shell tput sgr0)
# DEP_COMMAND := "command"
# DEP_FILE := "file"
MODULE := "huntag"
MODULE_PARAMS := ""

# Parse version string and create new version. Originally from: https://github.com/mittelholcz/contextfun
# Variable is empty in Travis-CI if not git tag present
TRAVIS_TAG ?= ""
OLDVER := $$(grep -P -o "(?<=__version__ = ')[^']+" $(MODULE)/version.py)

MAJOR := $$(echo $(OLDVER) | sed -r s"/([0-9]+)\.([0-9]+)\.([0-9]+)/\1/")
MINOR := $$(echo $(OLDVER) | sed -r s"/([0-9]+)\.([0-9]+)\.([0-9]+)/\2/")
PATCH := $$(echo $(OLDVER) | sed -r s"/([0-9]+)\.([0-9]+)\.([0-9]+)/\3/")

NEWMAJORVER="$$(( $(MAJOR)+1 )).0.0"
NEWMINORVER="$(MAJOR).$$(( $(MINOR)+1 )).0"
NEWPATCHVER="$(MAJOR).$(MINOR).$$(( $(PATCH)+1 ))"

all:
	@echo "See Makefile for possible targets!"

# extra:
# 	# Do extra stuff (e.g. compiling, downloading) before building the package

# clean-extra:
# 	rm -rf extra stuff

# install-dep-packages:
# 	# Install packages in Aptfile
# 	sudo -E apt-get update
# 	sudo -E apt-get -yq --no-install-suggests --no-install-recommends $(travis_apt_get_options) install `cat Aptfile`

# check:
# 	# Check for file or command
# 	@test -f ${DEP_FILE} >/dev/null 2>&1 || \
# 		 { echo >&2 "File \`${DEP_FILE}\` could not be found!"; exit 1; }
# 	@command -v ${DEP_COMMAND} >/dev/null 2>&1 || { echo >&2 "Command \`${DEP_COMMAND}\`could not be found!"; exit 1; }

dist/*.whl dist/*.tar.gz: # check extra
	@echo "Building package..."
	python3 setup.py sdist bdist_wheel

build: dist/*.whl dist/*.tar.gz

install-user: build
	@echo "Installing package to user..."
	pip3 install dist/*.whl

test-train:
	@echo "Running train tests..."
	# train
	time (cd /tmp && python3 -m ${MODULE} train --model=testMNP --config-file=configs/maxnp.szeged.emmorph.yaml \
						--gold-tag-field gold -i $(DIR)/tests/test.maxnp.emmorph 2>&1 | head -n100)
	@echo "$(green)Test OK$(sgr0)"
	time (cd /tmp && python3 -m ${MODULE} train --model=testNER --config-file=configs/ner.szeged.emmorph.yaml \
						--gold-tag-field gold -i $(DIR)/tests/test.ner.emmorph 2>&1 | head -n100)
	@echo "$(green)Test OK$(sgr0)"
	# train, featurize (for crfsuite)
	time (cd /tmp && python3 -m ${MODULE} train-featurize --model=testMNP --config-file=configs/maxnp.szeged.emmorph.yaml \
						--gold-tag-field gold -i $(DIR)/tests/test.maxnp.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.maxnp.CRFsuite.train 2>&1 | head -n100)
	@echo "$(green)Test OK$(sgr0)"
	time (cd /tmp && python3 -m ${MODULE} train-featurize --model=testNER --config-file=configs/ner.szeged.emmorph.yaml \
						--gold-tag-field gold -i $(DIR)/tests/test.ner.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.ner.CRFsuite.train 2>&1 | head -n100)
	@echo "$(green)Test OK$(sgr0)"
	# most-informative-features
	time (cd /tmp && python3 -m ${MODULE} most-informative-features --model=testMNP \
						--config-file=configs/maxnp.szeged.emmorph.yaml -i $(DIR)/tests/test.maxnp.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.maxnp.mostInformativeFeatures 2>&1 | head -n100)
	@echo "$(green)Test OK$(sgr0)"
	time (cd /tmp && python3 -m ${MODULE} most-informative-features --model=testNER \
						--config-file=configs/ner.szeged.emmorph.yaml -i $(DIR)/tests/test.ner.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.ner.mostInformativeFeatures 2>&1 | head -n100)
	@echo "$(green)Test OK$(sgr0)"
	# transmodel-train
	time (cd /tmp && python3 -m ${MODULE} transmodel-train --model=testMNP --config-file=configs/maxnp.szeged.emmorph.yaml \
						--gold-tag-field gold -i $(DIR)/tests/test.maxnp.emmorph 2>&1 | head -n100)
						# --trans-model-order [2 or 3, default: 3]
	@echo "$(green)Test OK$(sgr0)"
	time (cd /tmp && python3 -m ${MODULE} transmodel-train --model=testNER --config-file=configs/ner.szeged.emmorph.yaml \
						--gold-tag-field gold -i $(DIR)/tests/test.ner.emmorph 2>&1 | head -n100)
						# --trans-model-order [2 or 3, default: 3]
	@echo "$(green)Test OK$(sgr0)"

test-eval:
	@echo "Running eval tests..."
	# tag
	time (cd /tmp && python3 -m ${MODULE} tag --model=models/maxnp.szeged.emmorph \
						--config-file=configs/maxnp.szeged.emmorph.yaml --label-tag-field NP-BIO \
						-i $(DIR)/tests/test.maxnp.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.maxnp.tag 2>&1 | head -n100)
	@echo "$(green)Test OK$(sgr0)"
	time (cd /tmp && python3 -m ${MODULE} tag --model=models/ner.szeged.emmorph --config-file=configs/ner.szeged.emmorph.yaml \
						--label-tag-field NER-BIO -i $(DIR)/tests/test.ner.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.ner.tag 2>&1 | head -n100)
	@echo "$(green)Test OK$(sgr0)"
	# tag, featurize (for crfsuite)
	time (cd /tmp && python3 -m ${MODULE} tag-featurize --model=models/maxnp.szeged.emmorph \
						--config-file=configs/maxnp.szeged.emmorph.yaml -i $(DIR)/tests/test.maxnp.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.maxnp.CRFsuite.tag 2>&1 | head -n100)
	@echo "$(green)Test OK$(sgr0)"
	time (cd /tmp && python3 -m ${MODULE} tag-featurize --model=models/ner.szeged.emmorph \
						--config-file=configs/ner.szeged.emmorph.yaml -i $(DIR)/tests/test.ner.emmorph | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.ner.CRFsuite.tag 2>&1 | head -n100)
	@echo "$(green)Test OK$(sgr0)"
	# tag FeatureWeights
	time (cd /tmp && python3 -m ${MODULE} print-weights -w 100 --model=models/maxnp.szeged.emmorph \
						--config-file=configs/maxnp.szeged.emmorph.yaml | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.maxnp.modelWeights 2>&1 | head -n100)
	@echo "$(green)Test OK$(sgr0)"
	time (cd /tmp && python3 -m ${MODULE} print-weights -w 100 --model=models/ner.szeged.emmorph \
						--config-file=configs/ner.szeged.emmorph.yaml | \
						diff -sy --suppress-common-lines - $(DIR)/tests/test.ner.modelWeights 2>&1 | head -n100)
	@echo "$(green)Test OK$(sgr0)"

test: test-train test-eval

install-user-test: install-user test
	@echo "$(green)The test was completed successfully!$(sgr0)"

check-version:
	@echo "Comparing GIT TAG (\"$(TRAVIS_TAG)\") with pacakge version (\"v$(OLDVER)\")..."
	 @[[ "$(TRAVIS_TAG)" == "v$(OLDVER)" || "$(TRAVIS_TAG)" == "" ]] && \
	  echo "$(green)OK!$(sgr0)" || \
	  (echo "$(red)Versions do not match!$(sgr0)" && exit 1)

ci-test: install-user-test check-version

uninstall:
	@echo "Uninstalling..."
	pip3 uninstall -y ${MODULE}

install-user-test-uninstall: install-user-test uninstall

clean: # clean-extra
	rm -rf dist/ build/ ${MODULE}.egg-info/

clean-build: clean build

# Do actual release with new version. Originally from: https://github.com/mittelholcz/contextfun
release-major:
	@make -s __release NEWVER=$(NEWMAJORVER)
.PHONY: release-major


release-minor:
	@make -s __release NEWVER=$(NEWMINORVER)
.PHONY: release-minor


release-patch:
	@make -s __release NEWVER=$(NEWPATCHVER)
.PHONY: release-patch


__release:
	@if [[ -z "$(NEWVER)" ]] ; then \
		echo 'Do not call this target!' ; \
		echo 'Use "release-major", "release-minor" or "release-patch"!' ; \
		exit 1 ; \
		fi
	@if [[ $$(git status --porcelain) ]] ; then \
		echo 'Working dir is dirty!' ; \
		exit 1 ; \
		fi
	@echo "NEW VERSION: $(NEWVER)"
	@make clean uninstall install-user-test-uninstall
	@sed -i -r "s/__version__ = '$(OLDVER)'/__version__ = '$(NEWVER)'/" $(MODULE)/version.py
	@make check-version
	@git add $(MODULE)/version.py
	@git commit -m "Release $(NEWVER)"
	@git tag -a "v$(NEWVER)" -m "Release $(NEWVER)"
	@git push --tags
.PHONY: __release
