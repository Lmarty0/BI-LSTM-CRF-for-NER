#!/bin/bash
name=train

if [ $1 = $name ]
        then
                python bi-lstm-crf_train.py --training_mode train --training_file $2 --validation_file $3
        else
                python bi-lstm-crf_train.py --training_mode test --testing_file $2 --output_file $3
fi