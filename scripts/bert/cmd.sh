#!/bin/bash

# This script launches the following training jobs
# 1) BERT pre-train phase 1 (with seq-len = 128)
# 2) BERT pre-train phase 2 (with seq-len = 512). This requires the checkpoint from (1)
# 3) BERT fine-tune on SQuAD. This requires the checkpoint from (2).

export DATA_HOME=~/mxnet-data/bert-pretraining/datasets
export SYNTHETIC="${SYNTHETIC:-0}"

#export DATA="${DATA:-$DATA_HOME/book-corpus/book-corpus-large-split/*.train,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.train}"
#export DATAEVAL="${DATAEVAL:-$DATA_HOME/book-corpus/book-corpus-large-split/*.dev,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.dev}"
#export DATAPHASE2="${DATAPHASE2:-$DATA_HOME/book-corpus/book-corpus-large-split/*.train,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.train}"
export DATA="${DATA:-/fsx/datasets/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_512_shard//books_wiki_en_corpus_train/}"
export DATAEVAL="${DATAEVAL:-/fsx/datasets/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_512_shard//books_wiki_en_corpus_test/}"

export RAW="${RAW:-1}"
export EVALRAW="${EVALRAW:-0}"
export DTYPE=float16

mkdir -p $CKPTDIR
echo "==========================================================" >> $CKPTDIR/cmd.sh
cat cmd.sh >> $CKPTDIR/cmd.sh
echo "==========================================================" >> $CKPTDIR/cmd.sh

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
