# This script launches the following training jobs
# 1) BERT pre-train phase 1 (with seq-len = 128)
# 2) BERT pre-train phase 2 (with seq-len = 512). This requires the checkpoint from (1)
# 3) BERT fine-tune on SQuAD. This requires the checkpoint from (2).
export DATA_HOME=~/mxnet-data/bert-pretraining/datasets

export DEBUG="${DEBUG:-1}"
export HOST="${HOST:-hosts_32}"
export NP="${NP:-8}"
export CKPTDIR="${CKPTDIR:-./test-ckpt}"
export OPTIMIZER="${OPTIMIZER:-lamb2}"
export COMPLETE_TRAIN="${COMPLETE_TRAIN:-0}"
export DATA="${DATA:-$DATA_HOME/book-corpus/book-corpus-large-split/*.train,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.train}"
export DATAEVAL="${DATAEVAL:-$DATA_HOME/book-corpus/book-corpus-large-split/*.dev,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.dev}"
export DATAPHASE2="${DATAPHASE2:-$DATA_HOME/book-corpus/book-corpus-large-split/*.train,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.train}"
export NO_SHARD="${NO_SHARD:-0}"
export RAW="${RAW:-1}"
export EVALRAW="${EVALRAW:-0}"
export NUM_DATA_THREAD="${NUM_DATA_THREAD:-8}"
export SCALE_NORM="${SCALE_NORM:-1}"
export SKIP_GLOBAL_CLIP="${SKIP_GLOBAL_CLIP:-1}"
export PT_DECAY="${PT_DECAY:-1}"

# only used in a docker container
export USE_DOCKER=0
export OTHER_HOST=hosts_31
export DOCKER_IMAGE=haibinlin/worker_mxnet:c5fd6fc-1.5-cu90-79e6e8-79e6e8
export CLUSHUSER=ec2-user
export COMMIT=58435d04

export NCCLMINNRINGS=1
export TRUNCATE_NORM=1
export LAMB_BULK=60
export EPS_AFTER_SQRT=1
export NO_SHARD=0
export SKIP_STATE_LOADING=1
export REPEAT_SAMPLER=1
export FORCE_WD=0
export USE_PROJ=0
export DTYPE=float16
export MODEL=bert_24_1024_16
export CKPTINTERVAL=300000000
export HIERARCHICAL=0
export EVALINTERVAL=100000000
export NO_DROPOUT=0
export USE_BOUND=0
export ADJUST_BOUND=0
export WINDOW_SIZE=2000

mkdir -p $CKPTDIR
echo "==========================================================" >> $CKPTDIR/cmd.sh
cat cmd.sh >> $CKPTDIR/cmd.sh
echo "==========================================================" >> $CKPTDIR/cmd.sh

if [ "$USE_DOCKER" = "1" ]; then
    export PORT=12451
    bash clush-hvd.sh
else
    export PORT=22
fi

sleep 5

export OPTIONS='--verbose'
if [ "$DEBUG" = "1" ]; then
    export OPTIONS="$OPTIONS --synthetic_data"
    export NUMSTEPS=5000000000
    export LOGINTERVAL=5
else
    export NUMSTEPS=7038
    export LOGINTERVAL=10
fi
if [ "$RAW" = "1" ]; then
    export OPTIONS="$OPTIONS --raw"
fi
if [ "$EVALRAW" = "0" ]; then
    export OPTIONS="$OPTIONS --eval_use_npz"
fi

# export OPTIONS="$OPTIONS --start_step $NUMSTEPS"

#################################################################
# 1) BERT pre-train phase 1 (with seq-len = 128)
if [ "$NP" = "1" ]; then
    BS=64 ACC=1 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.0001 WARMUP_RATIO=0.2 bash mul-hvd.sh
elif [ "$NP" = "8" ]; then
    BS=512 ACC=1 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
elif [ "$NP" = "256" ]; then
    BS=65536 ACC=4 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.006 WARMUP_RATIO=0.2843 bash mul-hvd.sh
    echo 'DONE phase1'
fi
#################################################################


#################################################################
# 2) BERT pre-train phase 2 (with seq-len = 512). This requires the checkpoint from (1)
if [ "$COMPLETE_TRAIN" = "0" ]; then
    # skip phase 2 if COMPLETE_TRAIN = 0
    exit
fi
if [ "$USE_DOCKER" = "1" ]; then
    export PORT=12452
    bash clush-hvd.sh
else
    export PORT=22
fi

sleep 5
if [ "$DEBUG" = "1" ]; then
    export LOGINTERVAL=1
    export OPTIONS="--synthetic_data --verbose --phase2 --phase1_num_steps=$NUMSTEPS --start_step=$NUMSTEPS"
    export NUMSTEPS=3
else
    export LOGINTERVAL=10
    export OPTIONS="--phase2 --phase1_num_steps=$NUMSTEPS --start_step=$NUMSTEPS"
    export NUMSTEPS=1563
fi

export DATA=$DATAPHASE2
if [ "$NP" = "1" ]; then
    BS=8 ACC=1 MAX_SEQ_LENGTH=512 MAX_PREDICTIONS_PER_SEQ=80 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
elif [ "$NP" = "256" ]; then
    echo "$DATA"
    BS=32768 ACC=16 MAX_SEQ_LENGTH=512 MAX_PREDICTIONS_PER_SEQ=80 LR=0.004 WARMUP_RATIO=0.128 bash mul-hvd.sh
    echo 'DONE phase2'
fi


#################################################################


#################################################################
# 3) BERT fine-tune on SQuAD. This requires the checkpoint from (2).
STEP_FORMATTED=$(printf "%07d" $NUMSTEPS)
python3 finetune_squad.py --bert_model bert_24_1024_16 --pretrained_bert_parameters $CKPTDIR/$STEP_FORMATTED.params --output_dir $CKPTDIR --optimizer adam --accumulate 3 --batch_size 8 --lr 3e-5 --epochs 2 --gpu 0,1,2,3,4,5,6,7
#################################################################
