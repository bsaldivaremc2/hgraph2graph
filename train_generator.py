import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import sys
import gc


import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm
import traceback
import numpy as np
import logging
import os
from datetime import datetime


from hgraph import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model', default=None)
parser.add_argument('--seed', type=int, default=7)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--step_beta', type=float, default=0.001)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=25000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)
parser.add_argument('--log_name', type=str, default="train_log")
#parser.add_argument('--encoder_device', type=str, default='cuda:0')
#parser.add_argument('--decoder_device', type=str, default='cuda:1')


args = parser.parse_args()
print(args)
print("Setting logger")
time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
_log_file = "{}_{}.log".format(args.log_name,time_now)
logging.basicConfig(filename=_log_file, filemode='a', format='%(name)s - %(levelname)s - %(message)s',level=10)
def log_msg(msg):
    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    _m = "{}: {}.".format(time_now,msg)
    print(_m)
    logging.info(_m)
#
log_msg("Finished args parsing")

torch.manual_seed(args.seed)
random.seed(args.seed)

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)
log_msg("Finished vocab parsing")
model = HierVAE(args).cuda()
#model = HierVAE(args).to('cuda:0')
#model = HierVAE(args).to('cuda:1')

#model = HierVAE(args)
#if torch.cuda.device_count()>1:
#    model= nn.DataParallel(model).cuda()
#model.to("cuda")
#device = torch.device("cuda")
#model.cuda()
log_msg("Finished loading model")
m = "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)
print(m)
log_msg(m)

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

if args.load_model:
    log_msg('continuing from checkpoint ' + args.load_model)
    model_state, optimizer_state, total_step, beta = torch.load(args.load_model)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
else:
    total_step = beta = 0

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

meters = np.zeros(6)
ref_type = type(torch.tensor([0]))
m="Starting epoch"
print(m)
log_msg(m)
for epoch in range(args.epoch):
    dataset = DataFolder(args.train, args.batch_size)
    m="Loaded dataset per epoch"
    print(m)
    log_msg(m)
    _nbatch = 0
    _tbatch = len(dataset)

    for batch in tqdm(dataset):
        total_step += 1
        model.zero_grad()
        loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        #meters = meters + np.array([kl_div, loss.item(), wacc * 100, iacc * 100, tacc * 100, sacc * 100]) 
        #Change because of : https://github.com/wengong-jin/hgraph2graph/issues/40
        ms = [kl_div,loss.item(),wacc,iacc,tacc,sacc]
        msf= [1,1,100,100,100,100]
        for _i,_m_mf in enumerate(zip(ms,msf)):
            _m,_mf = _m_mf[0],_m_mf[1]
            if type(_m)==ref_type:
                _m = _m.cpu().numpy()
            ms[_i] = _m * _mf
        umeters = np.array(ms)
        meters = meters + umeters
        #meters = meters + np.array([kl_div, loss.item(), wacc.cpu() * 100, iacc.cpu() * 100, tacc.cpu() * 100, sacc.cpu() * 100])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            m="[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model))
            #print(m)
            log_msg(m)
            _batch_pos="Batch {}/{}".format(_nbatch,_tbatch-1)
            m=_batch_pos
            #print(m)
            log_msg(m)
            m=_batch_pos+" torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)
            #print(m)
            log_msg(m)
            m=_batch_pos+" torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024)
            #print(m)
            log_msg(m)
            m=_batch_pos+" torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024)
            #print(m)
            log_msg(m)
            sys.stdout.flush()
            meters *= 0
        
        if total_step % args.save_iter == 0:
            ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
            torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{total_step}"))

        if total_step % args.anneal_iter == 0:
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

        if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
            beta = min(args.max_beta, beta + args.step_beta)
        gc.collect() #Added
        _nbatch+=1
        del  loss, kl_div, wacc, iacc, tacc, sacc
        torch.cuda.empty_cache()
    log_msg("Finished epoch")
log_msg("Finished epochs. Done")

