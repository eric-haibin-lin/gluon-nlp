# DATA='/home/ec2-user/dataset/generated-enwiki-feb-uncased-py3-512/train/part-0/part-*.npz'
DATA='/home/ec2-user/dataset/generated-*-512/train/part-*/part-*.npz'
DATAEVAL='/home/ec2-user/dataset/generated-*-512/dev/part-*/part-*.npz'

DATA='/home/ec2-user/dataset/synthetic-book-feb-uncased-py3-128-partial/train/part-0/part-00*.npz'
DATAEVAL='/home/ec2-user/dataset/synthetic-book-feb-uncased-py3-128-partial/train/part-0/part-00*.npz'

pkill python
mpirun -np 8 --mca plm_rsh_agent "ssh -q -o StrictHostKeyChecking=no" \
       -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo \
       --map-by ppr:4:socket -x NCCL_MIN_NRINGS=16 -x NCCL_DEBUG=INFO \
       -x EPS_AFTER_SQRT=1 -x LAMB_BULK=30 \
       python3 run_pretraining_hvd.py \
       --batch_size 32 --accumulate 64 --lr 0.00354 \
       --data "$DATA" --data_eval "$DATAEVAL" \
       --warmup_ratio 0.1 --num_steps 28125 \
       --log_interval=2 --ckpt_dir large-lamb-16k.log \
       --optimizer lamb --model bert_24_1024_16 \
       --ckpt_interval 2500000000 --num_buckets 1 --dtype float16 2>&1 | tee -a large-lamb-16k.log
