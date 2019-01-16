PYTHONPATH=~/gluon-nlp/src/ python large_word_language_model.py --gpus 0 --clip=1  --log-interval 10 --sce

# eval
PYTHONPATH=~/gluon-nlp/src/ python large_word_language_model.py --gpus 0 --eval-only --batch-size=1 --eval-checkpoint model.params.00.000001000


# from pretrained
PYTHONPATH=~/gluon-nlp/src/ python large_word_language_model.py --gpus 0 --clip=1  --log-interval 10 --sce --pretrained --save from_pretrained.params

# eval
PYTHONPATH=~/gluon-nlp/src/ python large_word_language_model.py --gpus 0 --eval-only --batch-size=1 --save from_pretrained.params

# Loaded parameters from checkpoint from_pretrained.params.00
# Evaluation batch 1000: test loss 6.722917557, test ppl 831.239164554, throughput = 44.65 samples/s
# Evaluation batch 2000: test loss 6.703955173, test ppl 815.625393932, throughput = 49.78 samples/s
# Evaluation batch 3000: test loss 6.705263615, test ppl 816.693290257, throughput = 50.03 samples/s
# Evaluation batch 4000: test loss 6.701444149, test ppl 813.579907813, throughput = 49.13 samples/s
# Evaluation batch 5000: test loss 6.707398415, test ppl 818.438629371, throughput = 49.82 samples/s
# Evaluation batch 6000: test loss 6.702683449, test ppl 814.588802241, throughput = 48.84 samples/s
# Evaluation batch 7000: test loss 6.704124928, test ppl 815.763861380, throughput = 49.61 samples/s
# [Epoch 0] test loss 6.70, test ppl 814.28
# Epoch 0 took 163.64 seconds.
# Loaded parameters from checkpoint from_pretrained.params.01
# Evaluation batch 1000: test loss 5.801864624, test ppl 330.916018958, throughput = 46.28 samples/s
# Evaluation batch 2000: test loss 5.758562088, test ppl 316.892337868, throughput = 49.75 samples/s
# Evaluation batch 3000: test loss 5.746797085, test ppl 313.185944116, throughput = 49.47 samples/s
# Evaluation batch 4000: test loss 5.755842209, test ppl 316.031600087, throughput = 49.66 samples/s
# Evaluation batch 5000: test loss 5.752637386, test ppl 315.020396122, throughput = 48.73 samples/s
# Evaluation batch 6000: test loss 5.757311344, test ppl 316.496234483, throughput = 49.08 samples/s
# Evaluation batch 7000: test loss 5.754302502, test ppl 315.545378378, throughput = 49.49 samples/s
# [Epoch 1] test loss 5.75, test ppl 315.62
# Epoch 1 took 163.04 seconds.