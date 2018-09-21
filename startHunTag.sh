#!/bin/bash

#train
cat input.txt | python3 huntag_main.py train --model=modelName --config-file=configs/hunchunk.hunMIGE.yaml

#train to_crfsuite
cat input.txt | python3 huntag_main.py train-featurize --model=modelName --config-file=configs/hunchunk.hunMIGE.yaml > modelName.CRFsuite.train

#most-informative-features
cat input.txt | python3 huntag_main.py most-informative-features --model=modelName --config-file=configs/hunchunk.hunMIGE.yaml > modelName.mostInformativeFeatures

#transmodel-train:
cat input.txt | python3 huntag_main.py transmodel-train --model=modelName  # --trans-model-order [2 or 3, default: 3]

#tag
cat input.txt | python3 huntag_main.py tag --model=modelName --config-file=configs/hunchunk.hunMIGE.yaml > input.tag

#tag to_crfsuite
cat input.txt | python3 huntag_main.py tag-featurize --model=modelName --config-file=configs/hunchunk.hunMIGE.yaml > modelName.CRFsuite.tag

#tag FeatureWeights
cat input.txt | python3 huntag_main.py tag --printWeights 100 --model=modelName --config-file=configs/hunchunk.hunMIGE.yaml > modelName.modelWeights
