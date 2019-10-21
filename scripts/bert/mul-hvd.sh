# python -c 'import os; print(os.environ); import socket; print(socket.gethostname()); import mxnet as mx; import horovod.mxnet as hvd; print(mx); hvd.init(); print(hvd.rank())'

	    #--map-by ppr:4:socket:PE=4 \
	    #--synthetic_data --eval_use_npz \
pkill python

#mpirun -np $NP --hostfile hosts -display-allocation --allow-run-as-root \
#	    -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo \
#            --bind-to none \
#	    -x LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/lib \
#	    --mca plm_rsh_agent "ssh -q -o StrictHostKeyChecking=no -p $PORT" \
#	    -x NCCL_MIN_NRINGS=$NCCLMINNRINGS -x NCCL_DEBUG=INFO \
#	    -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
#	    -x HOROVOD_CYCLE_TIME=1 \
#	    -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=120 \
#	    -x MXNET_SAFE_ACCUMULATION=1 \
#	    --tag-output ompi_bind_DGX1.sh python run_pretraining.py \
#	    --data='/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train' \
#	    --data_eval='/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test' \
#	    --optimizer $OPTIMIZER \
#	    --warmup_ratio $WARMUP_RATIO \
#	    --num_steps $NUMSTEPS \
#	    --ckpt_interval $CKPTINTERVAL \
#	    --dtype $DTYPE \
#	    --ckpt_dir $CKPTDIR \
#	    --lr $LR \
#	    --total_batch_size $BS \
#	    --total_batch_size_eval $BS \
#	    --accumulate $ACC \
#	    --model $MODEL \
#	    --max_seq_length $MAX_SEQ_LENGTH \
#	    --max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
#	    --num_data_workers 4 \
#	    --no_compute_acc --raw \
#	    --verbose \
#	    --comm_backend horovod --log_interval $LOGINTERVAL

	    #--synthetic_data --eval_use_npz \
	    #--verbose \

mpirun -np $NP --hostfile $HOST -display-allocation --allow-run-as-root \
	    -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo \
            --bind-to none \
	    --mca plm_rsh_agent "ssh -q -o StrictHostKeyChecking=no -p $PORT" \
	    -x NCCL_MIN_NRINGS=$NCCLMINNRINGS -x NCCL_DEBUG=INFO \
	    -x HOROVOD_HIERARCHICAL_ALLREDUCE=$HIERARCHICAL \
	    -x HOROVOD_CYCLE_TIME=30 \
	    -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=120 \
	    -x MXNET_SAFE_ACCUMULATION=1 \
            -x NCCL_IB_DISABLE=1 \
            -x TRUNCATE_NORM=$TRUNCATE_NORM \
            -x LAMB_BULK=$LAMB_BULK \
            -x EPS_AFTER_SQRT=$EPS_AFTER_SQRT \
            -x NO_SHARD=$NO_SHARD \
	    --tag-output ompi_bind_DGX1.sh python3 run_pretraining.py \
	    --data='/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train' \
	    --data_eval='/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test' \
	    --optimizer $OPTIMIZER \
	    --warmup_ratio $WARMUP_RATIO \
	    --num_steps $NUMSTEPS \
	    --ckpt_interval $CKPTINTERVAL \
	    --dtype $DTYPE \
	    --ckpt_dir $CKPTDIR \
	    --lr $LR \
	    --total_batch_size $BS \
	    --total_batch_size_eval $BS \
	    --accumulate $ACC \
	    --model $MODEL \
	    --max_seq_length $MAX_SEQ_LENGTH \
	    --max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
	    --num_data_workers 4 \
            --eval_interval $EVALINTERVAL \
	    --no_compute_acc --raw \
	    --comm_backend horovod --log_interval $LOGINTERVAL $OPTIONS 2>&1 | tee -a mpi.log
