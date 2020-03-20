export ROOT_DIR="/fsx/datasets/ads"
export OUT_DIR="/fsx/datasets/ads/cleaned"
# python process.py --data $ROOT_DIR/BERT_training_impressed_asins_06_11_smaller_unencrypted/dev/*.txt, --out-dir $OUT_DIR/BERT_training_impressed_asins_06_11_smaller_unencrypted/dev
# python process.py --data $ROOT_DIR/BERT_training_impressed_asins_06_11_smaller_unencrypted/train/*.txt, --out-dir $OUT_DIR/BERT_training_impressed_asins_06_11_smaller_unencrypted/train
# python process.py --data $ROOT_DIR/BERT_training_3_months_snapshot_unencrypted/sampled/train/*.txt, --out-dir $OUT_DIR/BERT_training_3_months_snapshot_unencrypted/sampled/train
# python process.py --data $ROOT_DIR/BERT_training_3_months_snapshot_unencrypted/sampled/dev/*.txt, --out-dir $OUT_DIR/BERT_training_3_months_snapshot_unencrypted/sampled/dev
# python process.py --data $ROOT_DIR/BERT_training_3_months_snapshot_unencrypted/not_sampled/*.txt, --out-dir $OUT_DIR/BERT_training_3_months_snapshot_unencrypted/not_sampled

python process.py --data /fsx/datasets/ads/BERT_training_impressed_asins_06_11_smaller_unencrypted/dev/part-00199-99228f24-4824-48cd-aa8e-a6dc16b0736e-c000.txt, --out-dir . 
