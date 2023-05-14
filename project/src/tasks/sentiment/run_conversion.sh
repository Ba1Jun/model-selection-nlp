#!/bin/bash
DATA_PATH=project/resources/data/airline

# prepare and split data
python project/src/tasks/sentiment/convert.py $DATA_PATH/Tweets.csv $DATA_PATH/ -rs 4012