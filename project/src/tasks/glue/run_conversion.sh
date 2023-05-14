#!/bin/bash

DATA_PATHS=( project/resources/data/glue/bert project/resources/data/glue/roberta )
TASKS="mnli qnli rte"
SEPS=( "[SEP]" "</s>" )

for sep_idx in "${!SEPS[@]}"; do
  mkdir -p ${DATA_PATHS[$sep_idx]}
  python project/src/tasks/glue/convert.py -tasks $TASKS -output_path ${DATA_PATHS[$sep_idx]} -sep_token ${SEPS[$sep_idx]}
done