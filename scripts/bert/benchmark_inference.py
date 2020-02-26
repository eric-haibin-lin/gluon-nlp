import time
import argparse
import numpy as np
import mxnet as mx
import gluonnlp as nlp

parser = argparse.ArgumentParser(description='BERT pretraining example.')
parser.add_argument('--gpu', action='store_true', help='use GPU context')
parser.add_argument('--dtype', type=str, default='float32', help='data type')
args = parser.parse_args()
 
mx_ctx = mx.gpu() if args.gpu else mx.cpu()
seq_length = 128
num_classes = 2
dtype = args.dtype
get_model_params = {
    'name': "bert_12_768_12",
    'dataset_name': "book_corpus_wiki_en_uncased",
    'pretrained': False,
    'ctx': mx_ctx,
    'use_decoder': False,
    'use_classifier': False,
}

class BERTClassifier(mx.gluon.HybridBlock):
    def __init__(self, bert, num_classes=2, dropout=0.0,
                 prefix=None, params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.classifier = mx.gluon.nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(mx.gluon.nn.Dropout(rate=dropout))
            self.classifier.add(mx.gluon.nn.Dense(units=num_classes))

    def __call__(self, inputs, token_types, valid_length=None):
        return super(BERTClassifier, self).__call__(inputs, token_types, valid_length)

    def hybrid_forward(self, F, inputs, token_types, valid_length=None):
        _, pooler_out = self.bert(inputs, token_types, valid_length)
        return self.classifier(pooler_out)

bert, vocabulary = nlp.model.get_model(**get_model_params)
mx_model = BERTClassifier(bert, dropout=0.1, num_classes=num_classes)
mx_model.initialize(ctx=mx_ctx)
mx_model.hybridize(static_alloc=True)
 
inputs = np.random.uniform(size=(1, seq_length)).astype(dtype)
token_types = np.random.uniform(size=(1, seq_length)).astype(dtype)
valid_length = np.asarray([seq_length]).astype(dtype)
inputs_nd = mx.nd.array(inputs, ctx=mx_ctx)
token_types_nd = mx.nd.array(token_types, ctx=mx_ctx)
valid_length_nd = mx.nd.array(valid_length, ctx=mx_ctx)
 
mx_out = mx_model(inputs_nd, token_types_nd, valid_length_nd)
mx_out.wait_to_read()
 
min_repeat_ms = 2000
number = 10
while True:
    beg = time.time()
    for _ in range(number):
        mx_model(inputs_nd, token_types_nd, valid_length_nd).wait_to_read()
    end = time.time()
    lat = (end - beg) * 1e3
    if lat >= min_repeat_ms:
        break
    number = int(max(min_repeat_ms / (lat / number) + 1, number * 1.618))
print('mxnet mean lat: %.2f ms' % (lat / number))
