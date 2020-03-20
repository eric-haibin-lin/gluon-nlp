import mxnet as mx
import gluonnlp as nlp
import argparse
import collections
 
from gluonnlp.data import count_tokens
from bert_model import get_bert_model
 
parser = argparse.ArgumentParser(description='Get embeddings from BERT',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--convert', action='store_true', help='do conversion')
parser.add_argument('--old_params', type=str, help='old params')
parser.add_argument('--vocab', type=str, help='vocab')
parser.add_argument('--new_params', type=str, help='new params')
 
args = parser.parse_args()
 
 
class BERTForPretrain(mx.gluon.Block):
    def __init__(self, bert, mlm_loss, nsp_loss, vocab_size, prefix=None, params=None):
        super(BERTForPretrain, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        self.mlm_loss = mlm_loss
        self.nsp_loss = nsp_loss
        self._vocab_size = vocab_size
 
    def forward(self, input_id, masked_id, masked_position, masked_weight,
                next_sentence_label=None, segment_id=None, valid_length=None):
        # pylint: disable=arguments-differ
        """Predict with BERT for MLM and NSP. """
        num_masks = masked_weight.sum() + 1e-8
        valid_length = valid_length.reshape(-1)
        masked_id = masked_id.reshape(-1)
        _, _, classified, decoded = self.bert(input_id, segment_id, valid_length, masked_position)
        decoded = decoded.reshape((-1, self._vocab_size))
        ls1 = self.mlm_loss(decoded.astype('float32', copy=False),
                            masked_id, masked_weight.reshape((-1, 1)))
        ls2 = self.nsp_loss(classified.astype('float32', copy=False), next_sentence_label)
        ls1 = ls1.sum() / num_masks
        ls2 = ls2.mean()
        return classified, decoded, ls1, ls2
 
 
def interleave_qkv(query_weight, key_weight, value_weight, num_heads):
    # do something to reverse the weight
    query_weight_reverse = mx.nd.zeros_like(query_weight).reshape(shape=(num_heads, -1, 0), reverse=True)
    key_weight_reverse = mx.nd.zeros_like(key_weight).reshape(shape=(num_heads, -1, 0), reverse=True)
    value_weight_reverse = mx.nd.zeros_like(value_weight).reshape(shape=(num_heads, -1, 0), reverse=True)
 
    query_weight_split = query_weight.reshape(shape=(num_heads, -1, 0), reverse=True).split(num_outputs=num_heads,
                                                                                            axis=0)
    key_weight_split = key_weight.reshape(shape=(num_heads, -1, 0), reverse=True).split(num_outputs=num_heads, axis=0)
    value_weight_split = value_weight.reshape(shape=(num_heads, -1, 0), reverse=True).split(num_outputs=num_heads,
                                                                                            axis=0)
 
    query_idx = 0
    key_idx = 0
    value_idx = 0
    ws = [query_weight_reverse, key_weight_reverse, value_weight_reverse]
    i = int(0)
    ii = int(0)
    jj = int(0)
    while i < 3 * num_heads:
        if i < num_heads:
            source = query_weight_split
        elif i < num_heads * 2:
            source = key_weight_split
        else:
            source = value_weight_split
        x = i % num_heads
        mod = int(ii % (3 * num_heads))
        target = ws[mod // num_heads]
        xx = int(mod % num_heads)
        target[xx + jj][:] = source[x]
        i += 1
        ii += num_heads
        if i % 3 == 0:
            jj += 1
    for i in range(len(ws)):
        ws[i] = ws[i].reshape((768, -1))
    return tuple(ws)
 
 
# INPUTS
old_params = args.old_params  # '0300000.params.bert'
# sentencepiece = 'asin-unigram-32000-150M.model'
arr_dict = mx.nd.load(old_params)
print(arr_dict["encoder.transformer_cells.2.attention_cell.proj_key.weight"][0][:20])
prefix = 'encoder.'
pattern = 'transformer_cells.{}.attention_cell.proj_{}.weight'
num_layers = 12
num_heads = 12
 
# Conversion
for i in range(num_layers):
    q = prefix + pattern.format(i, 'query')
    k = prefix + pattern.format(i, 'key')
    v = prefix + pattern.format(i, 'value')
    interleave_q, interleave_k, interleave_v = interleave_qkv(arr_dict[q], arr_dict[k], arr_dict[v], num_heads)
    arr_dict[q][:] = interleave_q
    arr_dict[k][:] = interleave_k
    arr_dict[v][:] = interleave_v
 
new_params = args.new_params  # 'interleave_attn.params'
mx.nd.save(new_params, arr_dict)
 
# vocab = nlp.vocab.BERTVocab.from_sentencepiece(sentencepiece)
# vocab = nlp.vocab.BERTVocab.from_json(open(args.vocab).read())
 
# model, vocab = nlp.model.get_model('bert_12_768_12', dataset_name=None, vocab=vocab,
#                                    pretrained=False)  # , use_classifier=False, use_decoder=False)
 
model, vocab = get_bert_model('bert_12_768_12', vocab_file=args.vocab, pretrained=False, ctx=mx.cpu(0))
param = new_params if args.convert else old_params
nlp.utils.load_parameters(model, param, cast_dtype=True)  # , ignore_extra=True)
tokenizer = nlp.data.BERTTokenizer(vocab, lower=True)
transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=128, pair=False, pad=False)
sample = transform(['Hello world!'])
words, segments = mx.nd.array([sample[0]]), mx.nd.array([sample[2]])  # , mx.nd.array([sample[2]]);
print(words, segments, mx.nd.array([[1]]))
seq_encoding = model(words, segments, valid_length=None, masked_positions=mx.nd.array([[1]]))
if args.convert:
    print('inference with new params:')
    print(seq_encoding[0])
else:
    print('inference with old params:')
    print(seq_encoding[0])
