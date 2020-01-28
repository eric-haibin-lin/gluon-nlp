pkill python3

mpirun -np $BERT_CLUSTER_NP --hostfile $BERT_CLUSTER_HOST -display-allocation --allow-run-as-root \
	    -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo \
            --bind-to none \
            -x NCCL_SOCKET_IFNAME=eth0 \
            -x NCCL_IB_HCA=eth0 \
            -x FI_PROVIDER="efa" -x FI_EFA_TX_MIN_CREDITS=64 \
            -x LD_LIBRARY_PATH=$HOME/aws-ofi-nccl/install/lib/:$HOME/nccl/build/lib:/usr/local/cuda-10.0/lib64:/opt/amazon/efa/lib64:$LD_LIBRARY_PATH \
	    -x NCCL_MIN_NRINGS=$BERT_NCCL_MIN_NUM_RINGS \
            -x NCCL_DEBUG=VERSION \
	    -x HOROVOD_HIERARCHICAL_ALLREDUCE=$BERT_HVD_HIERARCHICAL \
	    -x HOROVOD_CYCLE_TIME=$BERT_HVD_CYCLE_TIME \
            -x HOROVOD_NUM_NCCL_STREAMS=2 \
	    -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=99999 \
	    -x MXNET_SAFE_ACCUMULATION=1 \
            -x MXNET_GPU_PARALLEL_RAND_COPY=$BERT_ENV_RAND_COPY \
            -x MXNET_GPU_WORKER_NTHREADS=$BERT_ENV_WORKER_NTHREAD \
            -x SKIP_STATE_LOADING=$BERT_ENV_SKIP_STATE_LOADING \
            -x MXNET_USE_FUSION=0 \
            -x MXNET_SEED=$BERT_ENV_MXNET_SEED \
            -x MXNET_OPTIMIZER_AGGREGATION_SIZE=$BERT_ENV_MXNET_OPTIMIZER_AGGREGATION_SIZE \
            -x NCCL_TREE_THRESHOLD=15360000 \
	    --tag-output \
            ompi_bind_DGX1.sh \
            python3 run_pretraining.py \
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
            --eval_interval $BERT_TRAIN_EVAL_INTERVAL \
	    --no_compute_acc \
	    --comm_backend horovod --log_interval $BERT_TRAIN_LOG_INTERVAL $OPTIONS 2>&1 | tee -a $BERT_TRAIN_CKPT_DIR/mpi.log
