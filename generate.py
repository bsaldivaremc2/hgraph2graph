import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse
from tqdm import tqdm

from hgraph import *
import rdkit
import traceback
import numpy as np
import logging
import os
from datetime import datetime


lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', required=True)

parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--nsample', type=int, default=10000)

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
parser.add_argument('--output_dir', type=str, default='generated')
parser.add_argument('--log_name', type=str, default="generate_log")


if __name__ == "__main__":
    args = parser.parse_args()
    print("Setting logger")
    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    _log_file = "{}_{}.log".format(args.log_name,time_now)
    logging.basicConfig(filename=_log_file, filemode='a', format='%(name)s - %(levelname)s - %(message)s',level=10)
    def log_msg(msg):
        time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        _m = "{}: {}.".format(time_now,msg)
        print(_m)
        logging.info(_m)
    log_msg("Finished args parsing")
    log_msg(args)
    #
    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
    args.vocab = PairVocab(vocab)
    log_msg("Vocab parsed")
    model = HierVAE(args).cuda()
    model.load_state_dict(torch.load(args.model)[0])
    model.eval()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    log_msg("Model loaded")
    os.makedirs(args.output_dir,exist_ok=True)
    output_file = os.path.join(args.output_dir,"generated_{}.txt".format(time_now))
    output_smiles = {}
    _batches = args.nsample // args.batch_size
    with torch.no_grad():
        #for _ in tqdm(range(args.nsample // args.batch_size)):
        for _batch in range(_batches):
            log_msg("Starting {}/{}".format(_batch,_batches-1))
            smiles_list = model.sample(args.batch_size, greedy=True)
            for _ in smiles_list:
                c = output_smiles.get(_,0)
                output_smiles[_]=c + 1
    log_msg("Saving results in {}".format(output_file))
    with open(output_file,"w") as f:
        for smile,count in output_smiles.items():
            _ = "{}\t{}\n".format(smile,count)
            f.write(_)
    log_msg("Done")
        

