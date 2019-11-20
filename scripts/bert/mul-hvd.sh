
#            python -c 'import os; import socket; import mxnet as mx; import horovod.mxnet as hvd; hvd.init(); print(socket.gethostname(), hvd.rank()); x = mx.nd.ones((1)).copyto(mx.gpu(hvd.local_rank())); hvd.allreduce_(x); print(x.asscalar()); import time; time.sleep(5)'
#exit
pkill python

	    #-x HOROVOD_TIMELINE=timeline.efa \

mpirun -np $NP --hostfile $HOST -display-allocation --allow-run-as-root \
	    -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo \
            --bind-to none \
	    --mca plm_rsh_agent "ssh -q -o StrictHostKeyChecking=no -p $PORT" \
            -x NCCL_SOCKET_IFNAME=eth0 \
            -x FI_PROVIDER="efa" -x FI_EFA_TX_MIN_CREDITS=64 \
            -x NCCL_IB_HCA=eth0 \
            -x NCCL_DEBUG=INFO \
	    -x NCCL_MIN_NRINGS=$NCCLMINNRINGS \
            -x HOROVOD_FUSION_THRESHOLD=268435456 \
	    -x HOROVOD_HIERARCHICAL_ALLREDUCE=$HIERARCHICAL \
            -x HOROVOD_NUM_NCCL_STREAMS=2 \
	    -x HOROVOD_CYCLE_TIME=30 \
	    -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=120 \
            -x NO_DROPOUT=$NO_DROPOUT \
            -x USE_BOUND=$USE_BOUND \
            -x USE_PROJ=$USE_PROJ \
            -x FORCE_WD=$FORCE_WD \
            -x LD_LIBRARY_PATH=$HOME/aws-ofi-nccl/install/lib/:$HOME/nccl/build/lib:/usr/local/cuda-10.0/lib64:/opt/amazon/efa/lib64:$LD_LIBRARY_PATH \
	    -x WINDOW_SIZE=$WINDOW_SIZE \
	    -x MXNET_SAFE_ACCUMULATION=1 \
            -x NCCL_TREE_THRESHOLD=15360000 \
            -x ADJUST_BOUND=$ADJUST_BOUND \
            -x TRUNCATE_NORM=$TRUNCATE_NORM \
            -x LAMB_BULK=$LAMB_BULK \
            -x EPS_AFTER_SQRT=$EPS_AFTER_SQRT \
            -x SKIP_GLOBAL_CLIP=$SKIP_GLOBAL_CLIP \
            -x PT_DECAY=$PT_DECAY \
            -x SKIP_STATE_LOADING=$SKIP_STATE_LOADING \
            -x REPEAT_SAMPLER=$REPEAT_SAMPLER \
            -x SCALE_NORM=$SCALE_NORM \
            -x MXNET_USE_FUSION=0 \
            -x FIX_BERT_ENCODER=1 \
            -x HD5=1 \
            -x NO_SHARD=$NO_SHARD \
	    --tag-output ompi_bind_DGX1.sh python3 run_pretraining.py \
	    --data="$DATA" \
	    --data_eval="$DATAEVAL" \
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
	    --num_data_workers $NUM_DATA_THREAD \
            --eval_interval $EVALINTERVAL \
	    --no_compute_acc \
	    --comm_backend horovod --log_interval $LOGINTERVAL $OPTIONS 2>&1 | tee -a $CKPTDIR/mpi.log
