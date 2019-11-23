# This script launches the following training jobs
# 1) BERT pre-train phase 1 (with seq-len = 128)
# 2) BERT pre-train phase 2 (with seq-len = 512). This requires the checkpoint from (1)
# 3) BERT fine-tune on SQuAD. This requires the checkpoint from (2).
export DATA_HOME=~/mxnet-data/bert-pretraining/datasets

export SYNTHETIC="${SYNTHETIC:-0}"
export HOST="${HOST:-hosts_64}"
export NP="${NP:-512}"
export CKPTDIR="${CKPTDIR:-/fsx/test-ckpt}"
export OPTIMIZER="${OPTIMIZER:-lamb3}"
export COMPLETE_TRAIN="${COMPLETE_TRAIN:-1}"

#export DATA="${DATA:-$DATA_HOME/book-corpus/book-corpus-large-split/*.train,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.train}"
#export DATAEVAL="${DATAEVAL:-$DATA_HOME/book-corpus/book-corpus-large-split/*.dev,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.dev}"
#export DATAPHASE2="${DATAPHASE2:-$DATA_HOME/book-corpus/book-corpus-large-split/*.train,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.train}"
export DATA="${DATA:-/fsx/datasets/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_512_shard//books_wiki_en_corpus_train/}"
export DATAEVAL="${DATAEVAL:-/fsx/datasets/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_512_shard//books_wiki_en_corpus_test/}"
export DATAPHASE2="${DATAPHASE2:-/fsx/datasets/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_512_shard//books_wiki_en_corpus_train/}"
export DATAPHASE2EVAL="${DATAPHASE2EVAL:-/fsx/datasets/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_512_shard//books_wiki_en_corpus_test/}"

export NO_SHARD="${NO_SHARD:-0}"
export RAW="${RAW:-1}"
export EVALRAW="${EVALRAW:-0}"
export NUM_DATA_THREAD="${NUM_DATA_THREAD:-8}"
export SCALE_NORM="${SCALE_NORM:-0}"
export SKIP_GLOBAL_CLIP="${SKIP_GLOBAL_CLIP:-0}"
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
export SKIP_STATE_LOADING=1
export REPEAT_SAMPLER=1
export FORCE_WD=0
export USE_PROJ=0
export DTYPE=float16
export FP32_LN=0
export FP32_SM=0
export MODEL=bert_24_1024_16
export CKPTINTERVAL=300000000
export HIERARCHICAL=0
export EVALINTERVAL=100000000
export NO_DROPOUT=0
export USE_BOUND=0
export ADJUST_BOUND=0
export WINDOW_SIZE=2000
export USE_AMP=0
export RESCALE_FAC="0"
export MANUAL_ACC=0
export USE_SA=1
export HD5=0

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
export NUMSTEPS=7038
export LOGINTERVAL=10

if [ "$SYNTHETIC" = "1" ]; then
    export OPTIONS="$OPTIONS --synthetic_data"
else
    export HD5=1
fi

if [ "$RAW" = "1" ]; then
    export OPTIONS="$OPTIONS --raw"
fi
if [ "$EVALRAW" = "0" ]; then
    export OPTIONS="$OPTIONS --eval_use_npz"
fi


#export OPTIONS="$OPTIONS --start_step 15625" #$NUMSTEPS"

#################################################################
# 1) BERT pre-train phase 1 (with seq-len = 128)
if [ "$NP" = "256" ]; then
    #BS=65536 ACC=8 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.006 WARMUP_RATIO=0.2843 bash mul-hvd.sh
    LOGINTERVAL=10 NUMSTEPS=14063 BS=32768 ACC=4 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
    echo 'DONE phase1'
elif [ "$NP" = "512" ]; then
    #BS=65536 ACC=8 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.006 WARMUP_RATIO=0.2843 bash mul-hvd.sh

    export NUMSTEPS=14063
    export DTYPE='float32'
    LOGINTERVAL=10 BS=32768 ACC=2 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
    #export NUMSTEPS=7038
    #LOGINTERVAL=10 BS=65536 ACC=2 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.003 WARMUP_RATIO=0.45 bash mul-hvd.sh
    echo 'DONE phase1'
elif [ "$NP" = "1" ]; then
    export DTYPE='float32'
    #LOGINTERVAL=10 BS=256 ACC=1 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
    LOGINTERVAL=10 BS=8 ACC=1 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
elif [ "$NP" = "8" ]; then
    export DTYPE='float32'
    #LOGINTERVAL=10 BS=256 ACC=1 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
    LOGINTERVAL=10 BS=64 ACC=1 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
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

export LOGINTERVAL=10
export OPTIONS="--phase2 --phase1_num_steps=$NUMSTEPS --start_step=$NUMSTEPS"
export NUMSTEPS=1564

if [ "$SYNTHETIC" = "1" ]; then
    export OPTIONS="$OPTIONS --synthetic_data"
else
    export HD5=1
fi

export DATA=$DATAPHASE2
export DATAEVAL=$DATAPHASE2EVAL
if [ "$NP" = "512" ]; then
    DTYPE='float32'
    BS=32768 ACC=16 MAX_SEQ_LENGTH=512 MAX_PREDICTIONS_PER_SEQ=80 LR=0.004 WARMUP_RATIO=0.128 bash mul-hvd.sh
    #DTYPE='float16'
    #BS=32768 ACC=8 MAX_SEQ_LENGTH=512 MAX_PREDICTIONS_PER_SEQ=80 LR=0.004 WARMUP_RATIO=0.4 bash mul-hvd.sh
    echo 'DONE phase2'
elif [ "$NP" = "8" ]; then
    DTYPE='float32'
    BS=32 ACC=1 MAX_SEQ_LENGTH=512 MAX_PREDICTIONS_PER_SEQ=80 LR=0.004 WARMUP_RATIO=0.128 bash mul-hvd.sh
fi

#################################################################


#################################################################
# 3) BERT fine-tune on SQuAD. This requires the checkpoint from (2).
#STEP_FORMATTED=$(printf "%07d" $NUMSTEPS)
#python3 finetune_squad.py --bert_model bert_24_1024_16 --pretrained_bert_parameters $CKPTDIR/$STEP_FORMATTED.params --output_dir $CKPTDIR --optimizer adam --accumulate 3 --batch_size 8 --lr 3e-5 --epochs 2 --gpu 0,1,2,3,4,5,6,7
#################################################################
