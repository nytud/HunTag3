#!/bin/bash

#train
cat input.txt | python huntag.py train --model=modelName --config-file=configs/hunchunk.krPatt.cfg

#bigram-train:
cat input.txt | python huntag.py bigram-train --model=modelName

#tag
cat input.txt | python huntag.py tag --model=modelName --config-file=configs/hunchunk.krPatt.cfg
