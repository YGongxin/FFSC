from __future__ import print_function
import argparse
import torch
from solver import Solver
import os
import time
import numpy as np
from torch.autograd import Variable

time_start = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
parser.add_argument('--source', type=str, default='SNL', metavar='N',
                    help='source dataset,mice,')
parser.add_argument('--target', type=str, default='230BreaKHis200_bal', metavar='N',
                    help='target dataset:war24')
parser.add_argument('--max_epoch', type=int, default=30, metavar='N',  # epoch
                    help='how many epochs')
parser.add_argument('--save_epoch', type=int, default=27, metavar='N',
                    help='when to restore the model')
parser.add_argument('--model', type=str, default='mlm_lout_flaten', metavar='N',
                    help='name of the model')
parser.add_argument('--expl', type=str, default='param', metavar='N',
                    help='explain the adjustment')
parser.add_argument('--mmd_pt', type=float, default=0.01, metavar='S',
                    help='rate of dataset IB 0.00001, BI 0.01')  # =====
parser.add_argument('--lpp_pt', type=float, default=0.01, metavar='S',
                    help='rate of dataset IB 0.000001, BI 0.005')  # ===
parser.add_argument('--condition_pt', type=float, default=0.001, metavar='S',
                    help='rate of dataset IB 0.000001, BI 0.005')  # ===
parser.add_argument('--gama', type=float, default=0.9, metavar='S',
                    help='learning rate attenuation coefficient (default: 0.9)')
parser.add_argument('--fnc', type=str, default='no', metavar='N',
                    help='use fnc?')
parser.add_argument('--lpp_dim', type=int, default=512, metavar='S',
                    help='dimensionality reduction')
parser.add_argument('--rate', type=int, default=5, metavar='S',  # 5 or 13
                    help='rate of dataset')
parser.add_argument('--ts_rate', type=float, default=1, metavar='S',  # 224 0.5607 336  0.45  0.63
                    help='rate of target and source')
parser.add_argument('--batch_size_tra', type=int, default=21, metavar='N',  # 19,23
                    help='input batch size for training (default: 32)')
parser.add_argument('--batch_size_tes', type=int, default=21, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--tlabel', type=int, default=30, metavar='N',
                    help='number of target train label (default: 100)')
# parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--num_k', type=int, default=2, metavar='N',
                    help='hyper paremeter for generator update')

parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')

parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')
parser.add_argument('--seed', type=int, default=3, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num_workers', type=int, default=4, metavar='S',
                    help='the number of processes loaded with multiple processes')
parser.add_argument('--pin_memory', action='store_false', default=False,
                    help='locked page memory')
parser.add_argument('--use_abs_diff', action='store_true', default=False,
                    help='use absolute difference value as a measurement')
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args, '\n')


def main():
    # if not args.one_step:
    record_name = '%s_%s_%s_%s' % (args.source.split('/')[-1], args.target.split('/')[-1], args.model, args.tlabel)
    if not os.path.exists('record/%s' % record_name):
        os.mkdir('record/%s' % record_name)

    record_num = 0
    loss_all = []
    mmd1_loss = []
    lpps_loss = []
    mmd2_loss = []
    mmd3_loss = []
    lppt_loss = []
    con_loss = []
    cen1_loss = []
    cen2_loss = []
    loss_p1s = []
    loss_p2s = []
    list_accs = []
    F1_Score = []
    patch_accs = []
    record_train = 'record/%s/tsrate_%s_batch_tra_%s_batch_tes_%s_lr_%s_seed_%s_%s.txt' % (
    record_name, args.ts_rate, args.batch_size_tra, args.batch_size_tes, args.lr, args.seed, record_num)
    record_test = 'record/%s/tsrate_%s_batch_tra_%s_batch_tes_%s_lr_%s_seed_%s_%s_test.txt' % (
    record_name, args.ts_rate, args.batch_size_tra, args.batch_size_tes, args.lr, args.seed, record_num)
    # record_pro = 'record/%s/batch_tra_%s_batch_tes_%s_lr_%s_seed_%s_%s_probability.txt' % (record_name, args.batch_size_tra, args.batch_size_tes, args.lr, args.seed, record_num)
    while os.path.exists(record_train):
        record_num += 1
        record_train = 'record/%s/tsrate_%s_batch_tra_%s_batch_tes_%s_lr_%s_seed_%s_%s.txt' % (
        record_name, args.ts_rate, args.batch_size_tra, args.batch_size_tes, args.lr, args.seed, record_num)
        record_test = 'record/%s/tsrate_%s_batch_tra_%s_batch_tes_%s_lr_%s_seed_%s_%s_test.txt' % (
        record_name, args.ts_rate, args.batch_size_tra, args.batch_size_tes, args.lr, args.seed, record_num)
        # record_pro = 'record/%s/batch_tra_%s_batch_tes_%s_lr_%s_seed_%s_%s_probability.txt' % (record_name, args.batch_size_tra, args.batch_size_tes, args.lr, args.seed, record_num)

    with open(record_train, 'a') as record:
        record.write(
            '--source: %s\n--target: %s\n--model: %s\n--ts_rate: %s\n--rate: %s\n--num_k: %s\n--max_epoch: %s\n\n' % (
            args.source, args.target, args.model,
            args.ts_rate, args.rate, args.num_k, args.max_epoch))
    with open(record_test, 'a') as record:
        record.write(
            '--source: %s\n--target: %s\n--model: %s\n--ts_rate: %s\n--rate: %s\n--num_k: %s\n--max_epoch: %s\n\n' % (
            args.source, args.target, args.model,
            args.ts_rate, args.rate, args.num_k, args.max_epoch))
    solver = Solver(args, source=args.source, target=args.target, model_name=args.model, num_workers=args.num_workers,
                    lpp_dim=args.lpp_dim,
                    pin_memory=args.pin_memory, learning_rate=args.lr, batch_size_tra=args.batch_size_tra,
                    batch_size_tes=args.batch_size_tes,
                    num_k=args.num_k, all_use=args.all_use,
                    save_epoch=args.save_epoch,
                    record_train=record_train, record_test=record_test,
                    tlabel=args.tlabel, seed=args.seed, rate=args.rate,
                    ts_rate=args.ts_rate, gama=args.gama, mmd_pt=args.mmd_pt, lpp_pt=args.lpp_pt,
                    condition_pt=args.condition_pt,expl=args.expl,
                    fnc=args.fnc )


    # exit()
    count = 0
    # 初始化聚类中心
    cs_1 = Variable(torch.cuda.FloatTensor(2, 1152).fill_(0))  # 初始化 #每层聚类中心 （类别数，全连接层维度）
    ct_1 = Variable(torch.cuda.FloatTensor(2, 1152).fill_(0))  # 每一次训练的时候都初始化了聚类中心，没有满足聚类中心的迭代
    cs_2 = Variable(torch.cuda.FloatTensor(2, 512).fill_(0))
    ct_2 = Variable(torch.cuda.FloatTensor(2, 512).fill_(0))
    for t in range(args.max_epoch):
        loss,loss_p1,loss_p2,cs_1, ct_1, cs_2, ct_2 = solver.train(t,cs_1,ct_1,cs_2,ct_2,record_file=record_train)

        loss_all.append(loss.cpu().detach().numpy())
        list_acc, F1, patch_acc = solver.test(t, record_file=record_test)
        list_accs.append(list_acc)
        F1_Score.append(F1)
        patch_accs.append(patch_acc)
        time_end = time.time()
        with open(record_test, 'a') as record:
            record.write('Epoch: %s\t Totally cost: %s\n' % (t, time_end - time_start))

    with open(record_test, 'a') as record:
        record.write('Max accuracy epoch: %s\n' % (list_accs.index(max(list_accs))))
    print('Max accuracy epoch: %s\t Totally cost: %s' % (list_accs.index(max(list_accs)), time_end - time_start))





if __name__ == '__main__':
    main()
