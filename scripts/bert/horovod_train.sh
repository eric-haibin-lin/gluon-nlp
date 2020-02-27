source parse_yaml.sh
CONFIG=$(parse_yaml phase2.config)
set -ex
eval $CONFIG

#horovodrun --mpi-args="-x NCCL_DEBUG=info" --log-level DEBUG --verbose -np 16 -H 172.31.12.211:8,172.31.7.89:8 -p 2022 python3 hvd_test.py
#python3 hvd_test.py
#exit
mpirun -np $BERT_CLUSTER_NP --hostfile $BERT_CLUSTER_HOST.mpi -display-allocation --allow-run-as-root \
	-mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo \
        -mca plm_rsh_args "-p 2022" \
        --bind-to none \
        -x LD_LIBRARY_PATH=$HOME/aws-ofi-nccl/install/lib/:$HOME/nccl/build/lib:/usr/local/cuda-10.0/lib64:/opt/amazon/efa/lib64:$LD_LIBRARY_PATH \
        -x FI_PROVIDER="efa" -x FI_EFA_TX_MIN_CREDITS=64 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_HCA=eth0 \
	-x NCCL_MIN_NRINGS=$BERT_NCCL_MIN_NUM_RINGS \
        -x NCCL_DEBUG=VERSION \
	-x HOROVOD_CYCLE_TIME=$BERT_HVD_CYCLE_TIME \
        -x HOROVOD_NUM_NCCL_STREAMS=1 \
	-x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=99999 \
	-x MXNET_SAFE_ACCUMULATION=1 \
        -x NCCL_TREE_THRESHOLD=15360000 \
	--tag-output \
        ompi_bind_p3dn.sh \
        python3 run_pretraining.py \
	--data="$BERT_PHASE2_DATA" \
	--data_eval="$BERT_PHASE2_DATA_EVAL" \
	--optimizer $BERT_TRAIN_OPTIMIZER \
	--warmup_ratio $BERT_PHASE2_WARMUP_RATIO \
	--num_steps $BERT_PHASE2_NUM_STEPS \
	--ckpt_interval $BERT_TRAIN_CKPT_INTERVAL \
	--dtype $BERT_TRAIN_DTYPE \
	--ckpt_dir $BERT_TRAIN_CKPT_DIR \
	--lr $BERT_PHASE2_LR \
	--total_batch_size $BERT_PHASE2_BS \
	--total_batch_size_eval $BERT_PHASE2_BS \
	--accumulate $BERT_PHASE2_ACC \
	--model $BERT_TRAIN_MODEL \
	--max_seq_length $BERT_PHASE2_MAX_SEQ_LENGTH \
	--max_predictions_per_seq $BERT_PHASE2_MAX_PREDICTIONS_PER_SEQ \
        --eval_interval $BERT_TRAIN_EVAL_INTERVAL \
	--no_compute_acc --phase2 --phase1_num_steps 14076 \
	--comm_backend horovod --log_interval $BERT_TRAIN_LOG_INTERVAL $BERT_PHASE2_OPTIONS 2>&1 | tee -a ~/stdout.log
