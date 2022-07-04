from torch.nn import functional as F

from .mask import fill_masked


def flatten_cross_entropy(pred, target, ignore_id=None):
    target = target.flatten()
    pred = pred.view(len(target), -1)
    return F.cross_entropy(pred, target, ignore_index=ignore_id)


def masked_flatten_cross_entropy(pred, target, mask, ignore_id=-100):
    _target = target.clone()

    # Игнорируем неизвестные слова при подсчёте loss-а
    for i, _batch in enumerate(_target):
        for j, _seq in enumerate(_batch):
            if _target[i][j][1] != -1:
                _target[i][j][0] = ignore_id

    _target = fill_masked(_target[:, :, 0], ~mask, ignore_id)
    return flatten_cross_entropy(pred, _target, ignore_id)
