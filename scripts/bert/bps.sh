#!/bin/bash
pkill python

set -ex

export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=99999
export MXNET_SAFE_ACCUMULATION=1
export MXNET_GPU_PARALLEL_RAND_COPY=$BERT_ENV_RAND_COPY
export MXNET_GPU_WORKER_NTHREADS=$BERT_ENV_WORKER_NTHREAD
export HD5=$BERT_TRAIN_HD5_DATA
export NO_DROPOUT=$BERT_ENV_NO_DROPOUT
export DISABLE_CUDNN_DROPOUT=$BERT_ENV_NO_CUDNN_DROPOUT
export TRUNCATE_NORM=$BERT_ENV_TRUNCATE_NORM
export SHARE_SEED=$BERT_ENV_SHARE_SEED
export FP32_LN=$BERT_ENV_FP32_LN
export FP32_SM=$BERT_ENV_FP32_SM
export PT_DECAY=$BERT_ENV_PT_DECAY
export SKIP_STATE_LOADING=$BERT_ENV_SKIP_STATE_LOADING
export MXNET_USE_FUSION=$BERT_ENV_USE_FUSION
export USE_GELU=$BERT_ENV_USE_GELU
export SM_LENGTH=$BERT_ENV_SM_LENGTH
export USE_SA=$BERT_ENV_USE_SA
export MANUAL_ACC=$BERT_ENV_MANUAL_ACC
export USE_AMP=$BERT_ENV_USE_AMP
export SLOW_NORM=$BERT_ENV_SLOW_NORM
export MXNET_SEED=$BERT_ENV_MXNET_SEED
export MXNET_OPTIMIZER_AGGREGATION_SIZE=$BERT_ENV_MXNET_OPTIMIZER_AGGREGATION_SIZE
export PER_STEP_NORM=$BERT_ENV_PER_STEP_NORM
export NO_SHARD=0
export SCALE_NORM=0
export USE_PROJ=0
export FORCE_WD=0
export WINDOW_SIZE=2000
# export NCCL_TREE_THRESHOLD=15360000
export ADJUST_BOUND=0
export USE_BOUND=0
export LAMB_BULK=60
export EPS_AFTER_SQRT=1
export SKIP_GLOBAL_CLIP=0
export RESCALE_FAC=0
export REPEAT_SAMPLER=1
export FIX_BERT_ENCODER=1
export SKIP_COMM=0
export NO_HYBRIDIZE=$BERT_ENV_NO_HYBRIDIZE

export NVIDIA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"

export BYTEPS_PARTITION_BYTES="${BYTEPS_PARTITION_BYTES:-4096000}"
export BYTEPS_NCCL_NUM_RINGS=1
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_NCCL_GROUP_SIZE="${BYTEPS_NCCL_GROUP_SIZE:-16}"
export BYTEPS_LOG_LEVEL=INFO # DEBUG

export BYTEPS_USE_HASH_KEY=1
export BPS_HOME="${BPS_HOME:-/usr/local/byteps}"
export DMLC_WORKER_ID="${DMLC_WORKER_ID:-0}"
export DMLC_NUM_WORKER="${DMLC_NUM_WORKER:-1}"
export DMLC_ROLE=worker
# export NCCL_MIN_NRINGS="${NCCL_MIN_NRINGS:-16}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD:-120}"
# export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD:-120}"
export MXNET_SAFE_ACCUMULATION="${MXNET_SAFE_ACCUMULATION:-1}"
export OPTIONS="${OPTIONS:---synthetic_data --eval_use_npz}"
export DATA="${DATA:-/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train}"
export DATA_EVAL="${DATA_EVAL:-/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test}"
export NUM_DATA_THREAD="${NUM_DATA_THREAD:-8}"
export SLOW_NORM=0
export BYTEPS_TRACE_ON=0
export BYTEPS_TRACE_START_STEP=21
export BYTEPS_TRACE_END_STEP=25
export USE_GELU=1
export BYTEPS_TRACE_DIR=~/bert_traces

echo $NVIDIA_VISIBLE_DEVICES
mkdir -p $BERT_TRAIN_CKPT_DIR

python3 -u $BPS_HOME/launcher/launch.py \
	python3 -u run_pretraining.py \
	    --data="$DATA" \
	    --data_eval="$DATA_EVAL" \
	    --optimizer $BERT_TRAIN_OPTIMIZER \
	    --warmup_ratio $WARMUP_RATIO \
	    --num_steps $NUM_STEPS \
	    --ckpt_interval $BERT_TRAIN_CKPT_INTERVAL \
	    --dtype $BERT_TRAIN_DTYPE \
	    --ckpt_dir $BERT_TRAIN_CKPT_DIR \
	    --lr $LR \
	    --total_batch_size $BS \
	    --total_batch_size_eval $BS \
	    --accumulate $ACC \
	    --model $BERT_TRAIN_MODEL \
	    --max_seq_length $MAX_SEQ_LENGTH \
	    --max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
	    --num_data_workers 8 \
            --eval_interval 100000000 \
	    --no_compute_acc \
	    --comm_backend byteps --log_interval $BERT_TRAIN_LOG_INTERVAL $OPTIONS 2>&1 | tee -a $BERT_TRAIN_CKPT_DIR/std.log.$DMLC_WORKER_ID
