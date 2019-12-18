#!/bin/bash
# =====================================================================
# This script launches the following training jobs:
# 1) BERT pre-train phase 1 (with seq-len = 128)
# 2) BERT pre-train phase 2 (with seq-len = 512). This requires the checkpoint from (1)
# =====================================================================
source parse_yaml.sh
export CONTAINER_REGISTRY=$1
export CONFIG="${CONFIG:-configurations/default.yml}"
export BACKEND="${BACKEND:-horovod}"

PARSED_DEFAULT=$(parse_yaml configurations/default.yml)
PARSED_NEW=$(parse_yaml $CONFIG)
eval $PARSED_DEFAULT
eval $PARSED_NEW

set -ex

export BERT_TRAIN_CKPT_DIR="$BERT_TRAIN_CKPT_DIR/$CONTAINER_REGISTRY"
printenv | sort | grep BERT
mkdir -p $BERT_TRAIN_CKPT_DIR
cp $CONFIG $BERT_TRAIN_CKPT_DIR/

# =====================================================================
# phase 1
# =====================================================================
if [ "$BERT_RUN_PHASE1" = "1" ]; then
    export DATA=$BERT_PHASE1_DATA
    export DATA_EVAL=$BERT_PHASE1_DATA_EVAL
    export OPTIONS=$BERT_PHASE1_OPTIONS
    export NUM_STEPS=$BERT_PHASE1_NUM_STEPS
    export BS=$BERT_PHASE1_BS
    export ACC=$BERT_PHASE1_ACC
    export MAX_SEQ_LENGTH=$BERT_PHASE1_MAX_SEQ_LENGTH
    export MAX_PREDICTIONS_PER_SEQ=$BERT_PHASE1_MAX_PREDICTIONS_PER_SEQ
    export LR=$BERT_PHASE1_LR
    export WARMUP_RATIO=$BERT_PHASE1_WARMUP_RATIO
    if [ "$BACKEND" = 'horovod' ]; then
        bash mul-hvd.sh
    else
        bash bps.sh
    fi
    echo 'DONE PHASE1'
fi

# =====================================================================
# phase 2
# =====================================================================
if [ "$BERT_RUN_PHASE2" = "1" ]; then
    export DATA=$BERT_PHASE2_DATA
    export DATA_EVAL=$BERT_PHASE2_DATA_EVAL
    export OPTIONS=$BERT_PHASE2_OPTIONS
    export NUM_STEPS=$BERT_PHASE2_NUM_STEPS
    export BS=$BERT_PHASE2_BS
    export ACC=$BERT_PHASE2_ACC
    export MAX_SEQ_LENGTH=$BERT_PHASE2_MAX_SEQ_LENGTH
    export MAX_PREDICTIONS_PER_SEQ=$BERT_PHASE2_MAX_PREDICTIONS_PER_SEQ
    export LR=$BERT_PHASE2_LR
    export WARMUP_RATIO=$BERT_PHASE2_WARMUP_RATIO
    export OPTIONS="$OPTIONS --phase2 --phase1_num_steps=$BERT_PHASE1_NUM_STEPS --start_step=$BERT_PHASE1_NUM_STEPS"
    if [ "$BACKEND" = 'horovod' ]; then
        bash mul-hvd.sh
    else
        bash bps.sh
    fi
    echo 'DONE PHASE2'
fi
