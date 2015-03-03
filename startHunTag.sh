#!/bin/bash

#train
cat input.txt | python3 huntag.py train --model=modelName --config-file=configs/hunchunk.krPatt.cfg

#train toCRFsuite
cat input.txt | python3 huntag.py train --toCRFsuite --model=modelName --config-file=configs/hunchunk.krPatt.cfg > modelName.CRFsuite.train

#most-informative-features
cat input.txt | python3 huntag.py most-informative-features --model=modelName --config-file=configs/hunchunk.krPatt.cfg > modelName.mostInformativeFeatures

#bigram-train:
cat input.txt | python3 huntag.py bigram-train --model=modelName

#tag
cat input.txt | python3 huntag.py tag --model=modelName --config-file=configs/hunchunk.krPatt.cfg

#tag toCRFsuite
cat input.txt | python3 huntag.py tag --toCRFsuite --model=modelName --config-file=configs/hunchunk.krPatt.cfg > modelNam.CRFsuite.tag

#tag FeatureWeights
cat input.txt | python3 huntag.py tag --printWeights 100 --model=modelName --config-file=configs/hunchunk.krPatt.cfg > modelNam.modelWeights
