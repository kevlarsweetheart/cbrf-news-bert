from .record import Record
from .mask import mask_like


class Acc(Record):
    __attributes__ = ['correct', 'total']

    def __init__(self, correct=0, total=0):
        self.correct = correct
        self.total = total

    def add(self, other):
        self.correct += other.correct
        self.total += other.total

    @property
    def value(self):
        if not self.total:
            return 0
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0


class Mean(Record):
    __attributes__ = ['accum', 'count']

    def __init__(self, accum=0, count=0):
        self.accum = accum
        self.count = count

    def add(self, value):
        self.accum += value
        self.count += 1

    @property
    def value(self):
        if not self.count:
            return 0
        return self.accum / self.count

    def reset(self):
        self.accum = 0
        self.count = 0


class F1(Record):
    __attributes__ = ['prec', 'recall']

    def __init__(self, prec=None, recall=None):
        if not prec:
            prec = Acc()
        self.prec = prec
        if not recall:
            recall = Acc()
        self.recall = recall

    def add(self, other):
        self.prec.add(other.prec)
        self.recall.add(other.recall)

    @property
    def value(self):
        prec = self.prec.value
        recall = self.recall.value
        if not prec + recall:
            return 0
        return 2 * prec * recall / (prec + recall)

    def reset(self):
        self.prec.reset()
        self.recall.reset()


def topk_acc(pred, target, ks=(1, 2, 4, 8), mask=None):
    k = max(ks)
    pred = pred.topk(
        k,
        dim=-1,
        largest=True,
        sorted=True
    ).indices

    if mask is None:
        mask = mask_like(target[:, :, 0])

    _target = target[:, :, 0][mask]
    pred = pred[mask].view(-1, k)  # restore shape

    pred = pred.t()  # k x tests
    _target = _target.expand_as(pred)  # k x tests

    correct = (pred == _target)
    total = mask.sum().item()
    for k in ks:
        count = correct[:k].sum().item()
        yield Acc(count, total)


def acc(a, b, mask=None):
    if mask is None:
        mask = mask_like(a)

    a = a[mask]
    b = b[mask]
    correct = (a == b).sum().item()
    total = len(a)
    return Acc(correct, total)


class BatchScore(Record):
    __attributes__ = ['loss']


class ScoreMeter(Record):
    __attributes__ = ['loss']

    def extend(self, scores):
        for score in scores:
            self.add(score)


###########
#
#   MLM
#
#######


class MLMBatchScore(BatchScore):
    __attributes__ = ['loss', 'ks']

    def __init__(self, loss, ks):
        self.loss = loss
        self.ks = ks


class MLMScoreMeter(ScoreMeter):
    __attributes__ = ['loss', 'ks']

    def __init__(self, loss=None, ks=None):
        if not loss:
            loss = Mean()
        if not ks:
            ks = {}
        self.loss = loss
        self.ks = ks

    def add(self, score):
        self.loss.add(score.loss)
        for k, score in score.ks.items():
            if k not in self.ks:
                self.ks[k] = score
            else:
                self.ks[k].add(score)

    def reset(self):
        self.loss.reset()
        for k in self.ks:
            self.ks[k].reset()

    def write(self, board):
        board.add_scalar('01_loss', self.loss.value)
        for index, k in enumerate(self.ks, 2):
            key = '%02d_top%d' % (index, k)
            score = self.ks.get(k)
            if score:
                board.add_scalar(key, score.value)


def score_mlm_batch(batch, ks=(1, 2, 4, 8)):
    scores = ()
    if ks:
        scores = topk_acc(pred=batch.pred, target=batch.target.value, ks=ks, mask=batch.target.mask)
    return MLMBatchScore(
        batch.loss.item(),
        ks=dict(zip(ks, scores))
    )


def score_mlm_batches(batches):
    for batch in batches:
        yield score_mlm_batch(batch)
