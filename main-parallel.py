from bert.model.bert import (
    CBNewsBERTConfig,
    BERTEmbedding,
    BERTEncoder,
    BERTMLMHead,
    BERTMLM
)
from bert.score import (
    MLMScoreMeter,
    score_mlm_batch,
    score_mlm_batches
)
from bert.const import (
    MODEL_ENCODER, MODEL_EMB, MODEL_MLM,
    TRAIN_BOARD, TEST_BOARD,
    BOARD_NAME, RUNS_DIR
)
from bert.loss import masked_flatten_cross_entropy
from bert.board import TensorBoard

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

import pickle
import argparse
import os
from datetime import datetime


def every(step, period):
    return step > 0 and step % period == 0


def process_batch(model, criterion, batch):
    pred = model(batch.input)
    loss = criterion(pred, batch.target.value, batch.target.mask)
    return batch.processed(loss, pred)


def infer_batches(model, criterion, batches):
    training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            yield process_batch(model, criterion, batch)
    model.train(training)


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    config = CBNewsBERTConfig()
    emb = BERTEmbedding.from_config(config)
    encoder = BERTEncoder.from_config(config)
    head = BERTMLMHead(config.emb_dim, config.main_vocab_size + config.unknown_size + 1)
    model = BERTMLM(emb, encoder, head)

    emb.position.weight.requires_grad = False
    criterion = masked_flatten_cross_entropy
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    with open('train_batches256_encoded.pkl', 'rb') as infile:
        train_batches = pickle.load(infile)
    with open('test_batches256_encoded.pkl', 'rb') as infile:
        test_batches = pickle.load(infile)

    board = TensorBoard(BOARD_NAME, RUNS_DIR)
    train_board = board.section(TRAIN_BOARD)
    test_board = board.section(TEST_BOARD)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.999)

    train_meter = MLMScoreMeter()
    test_meter = MLMScoreMeter()

    accum_steps = 10
    log_steps = 16
    eval_steps = 32
    save_steps = eval_steps * 10

    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model.train()
    optimizer.zero_grad()
    start = datetime.now()

    inputs = torch.arange(end=len(train_batches))
    targets = torch.arange(end=len(train_batches))
    train_indexes = TensorDataset(inputs, targets)
    train_sampler = DistributedSampler(train_indexes, num_replicas=args.world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_indexes,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    for epoch in range(args.epochs):
        epoch_loss = 0
        for step, (ind, _) in enumerate(train_loader):
            batch = process_batch(model, criterion, train_batches[ind])
            batch.loss /= accum_steps

            batch.loss.backward()

            score = score_mlm_batch(batch, ks=())
            train_meter.add(score)

            if every(step, log_steps):
                train_meter.write(train_board)
                train_meter.reset()

            if every(step, accum_steps):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if every(step, eval_steps):
                    batches = infer_batches(model, criterion, test_batches)
                    scores = score_mlm_batches(batches)
                    test_meter.extend(scores)
                    test_meter.write(test_board)
                    test_meter.reset()

            if every(step, save_steps):
                model.emb.dump(MODEL_EMB)
                model.encoder.dump(MODEL_ENCODER)
                model.head.dump(MODEL_MLM)

            board.step()
            epoch_loss += batch.loss.item()

        epoch_loss /= len(train_batches)
        print("Epoch #{_epoch}".format(_epoch=epoch + 1))
        print("Mean batch loss:", epoch_loss)

    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8887'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()
