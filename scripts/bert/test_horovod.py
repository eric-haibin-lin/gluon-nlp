import mxnet as mx
print('mxnet import done')
import horovod.mxnet as hvd
print('hvd import done')
hvd.init()
print('hvd init done', hvd.size())
a = mx.nd.ones((1,), ctx=mx.gpu(hvd.local_rank()))
print('mxnet ndarray creation done')
b = hvd.allreduce(a, name='test')
print('hvd allreduce done')
print(b)
