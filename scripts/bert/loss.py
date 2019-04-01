from mxnet import gluon
from mxnet.gluon import loss

class DistillationSoftmaxCrossEntropyLoss(gluon.HybridBlock):
    """SoftmaxCrossEntrolyLoss with Teacher model prediction

    Parameters
    ----------
    temperature : float, default 1
        The temperature parameter to soften teacher prediction.
    hard_weight : float, default 0.5
        The weight for loss on the one-hot label.
    sparse_label : bool, default True
        Whether the one-hot label is sparse.
    """
    def __init__(self, temperature=20, hard_weight=0.5, sparse_label=True, **kwargs):
        super(DistillationSoftmaxCrossEntropyLoss, self).__init__(**kwargs)
        self._temperature = temperature
        self._hard_weight = hard_weight
        with self.name_scope():
            self.soft_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False, **kwargs)
            self.hard_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label, **kwargs)
    def hybrid_forward(self, F, output, label, soft_target, sample_weight=None):
        if self._hard_weight == 0:
            return (self._temperature ** 2) * self.soft_loss(output / self._temperature, soft_target, sample_weight)
        elif self._hard_weight == 1:
            return self.hard_loss(output, label, sample_weight)
        else:
            soft_loss = (self._temperature ** 2) * self.soft_loss(output / self._temperature, soft_target, sample_weight)
            hard_loss = self.hard_loss(output, label, sample_weight)
            return (1 - self._hard_weight) * soft_loss  + self._hard_weight * hard_loss
