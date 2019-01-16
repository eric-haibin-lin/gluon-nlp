PYTHONPATH=~/gluon-nlp/src/ python large_word_language_model.py --gpus 0 --clip=1  --log-interval 10 --sce

# eval
PYTHONPATH=~/gluon-nlp/src/ python large_word_language_model.py --gpus 0 --eval-only --batch-size=1 --add-prior --eval-checkpoint model.params.00.000001000
