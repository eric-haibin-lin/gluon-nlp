Run training with batch size = 2 on GPU 0:
```
MXNET_GPU_MEM_POOL_TYPE=Round python train_gnmt.py --src_lang en --tgt_lang vi --optimizer adam --lr 0.001 --lr_update_factor 0.5 --beam_size 10 --bucket_scheme exp  --num_hidden 512 --save_dir gnmt_en_vi_l2_h512_beam10 --epochs 12 --log_interval=5 --batch_size 2 --gpu 0
```
To run training on CPU, you need to remove the `--gpu 0` option.
To run profiler, append `--profile`
