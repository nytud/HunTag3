#!/bin/bash

#train
cat test.maxnp.emmorph | python3 huntag_main.py train --model=modelName --config-file=configs/maxnp.szeged.emmorph.yaml --gold-tag-field 'NP-BIO'

#train, featurize (for crfsuite)
cat test.maxnp.emmorph | python3 huntag_main.py train-featurize --model=modelName --config-file=configs/maxnp.szeged.emmorph.yaml  --gold-tag-field 'NP-BIO' > modelName.CRFsuite.train

#most-informative-features
cat test.maxnp.emmorph | python3 huntag_main.py most-informative-features --model=modelName --config-file=configs/maxnp.szeged.emmorph.yaml --gold-tag-field 'NP-BIO' > modelName.mostInformativeFeatures

#transmodel-train:
cat test.maxnp.emmorph | python3 huntag_main.py transmodel-train --model=modelName --gold-tag-field 'NP-BIO'  # --trans-model-order [2 or 3, default: 3]

#tag
cat test.maxnp.emmorph | python3 huntag_main.py tag --model=modelName --config-file=configs/maxnp.szeged.emmorph.yaml --tag-field 'NP-BIO' > input.tag

#tag, featurize (for crfsuite)
cat test.maxnp.emmorph | python3 huntag_main.py tag-featurize --model=modelName --config-file=configs/maxnp.szeged.emmorph.yaml --tag-field 'NP-BIO' > modelName.CRFsuite.tag

#tag FeatureWeights
cat test.maxnp.emmorph | python3 huntag_main.py print-weights -w 100 --model=modelName --config-file=configs/maxnp.szeged.emmorph.yaml --tag-field 'NP-BIO' > modelName.modelWeights
