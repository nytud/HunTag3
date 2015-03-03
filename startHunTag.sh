#!/bin/bash

#train
cat input.txt | python huntag.py train --model=modelName.model --parameters="-s 0 -q" --config-file=configs/hunchunk.krPatt.cfg

#bigram-train:
cat input.txt | python huntag.py bigram-train --bigram-model=modelName.bigramModel --tag-field=2

#tag
cat input.txt | python huntag.py tag --model=modelName.model --bigram-model=modelName.bigramModel --config-file=configs/hunchunk.krPatt.cfg
