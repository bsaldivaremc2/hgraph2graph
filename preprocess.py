from multiprocessing import Pool
import math, random, sys
import pickle
import argparse
from functools import partial
import torch
import numpy
import traceback
import numpy as np
import logging
import os
from datetime import datetime


from hgraph import MolGraph, common_atom_vocab, PairVocab, Vocab
#from hgraph import MolGraph, PairVocab, Vocab
import rdkit

def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)

def tensorize_pair(mol_batch, vocab):
    x, y = zip(*mol_batch)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) #no need of order for x

def tensorize_cond(mol_batch, vocab):
    x, y, cond = zip(*mol_batch)
    cond = [map(int, c.split(',')) for c in cond]
    cond = numpy.array(cond)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) + (cond,) #no need of order for x

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', type=str, default='pair')
    parser.add_argument('--ncpu', type=int, default=8)
    parser.add_argument('--log_name', type=str, default="preproc_log")
    parser.add_argument('--output_dir', type=str, default="train_processed")
    parser.add_argument('--split_info_dir', type=str, default="molecules_split_info")
    args = parser.parse_args()
    #ADDED
    #
    print("Opening vocab")
    with open(args.vocab) as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    print("Pairing vocab")
    args.vocab = PairVocab(vocab, cuda=False)
    #
    print("Setting logger")
    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    _log_file = "{}_{}.log".format(args.log_name,time_now)
    logging.basicConfig(filename=_log_file, filemode='a', format='%(name)s - %(levelname)s - %(message)s',level=10)
    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info('Starting CPU pool creation. '+time_now)
    #
    os.makedirs(args.output_dir,exist_ok=True)
    os.makedirs(args.split_info_dir,exist_ok=True)
    logging.info('Storing in '+args.output_dir)
    #
    print("Creating CPU pool")
    pool = Pool(args.ncpu) 
    print("Pool created")
    random.seed(1)

    if args.mode == 'pair':
        #dataset contains molecule pairs
        print("Opening pair mode")
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:2] for line in f]

        random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize_pair, vocab = args.vocab)
        all_data = pool.map(func, batches)
        num_splits = max(len(all_data) // 1000, 1)

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'cond_pair':
        print("Opening cond pair mode")
        #dataset contains molecule pairs with conditions
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:3] for line in f]

        #random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize_cond, vocab = args.vocab)
        all_data = pool.map(func, batches)
        num_splits = max(len(all_data) // 1000, 1)

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'single':
        #dataset contains single molecules
        print("Opening single mode")
        print("Opening molecule list")
        with open(args.train) as f:
            complete_data = [line.strip("\r\n ").split()[0] for line in f]
        random.shuffle(complete_data)
        ###Spliting the data in splits before spliting, 
        split_factor = 1000
        split_step = split_factor * args.batch_size
        max_splits = int(np.ceil(len(complete_data)/split_step))
        msplits_digits = len(str(max_splits))
        print(max_splits,msplits_digits)
        def num_to_digit(inum):
            _ = str(inum)
            _l = len(_)
            lzeros=(msplits_digits-_l)*"0"
            return lzeros+_
        split_id = 0
        def proc_split(idata,ofile,msg=""):
            print("Generating batches")
            batches = [idata[i : i + args.batch_size] for i in range(0, len(idata), args.batch_size)]
            print("Tensorizing")
            func = partial(tensorize, vocab = args.vocab)
            print("Pooling")
            print("Pool created")
            all_data = pool.map(func, batches)
            print("Saving tensor pickle",ofile)
            with open(ofile, 'wb') as f:
                pickle.dump(all_data, f, pickle.HIGHEST_PROTOCOL)

        for start_split in range(0,len(complete_data),split_step):
            end_split = start_split+split_step
            split_code = num_to_digit(split_id)
            _of = "tensors-{}.pkl".format(split_code)
            _of = os.path.join(args.output_dir,_of)
            time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
            m = "Split: {}/{}. Start: {}. End:{}. {}".format(split_code,max_splits-1,start_split,end_split,time_now)
            #logging.warning('This will get logged to a file')
            print(m)
            logging.info(m)
            data = complete_data[start_split:end_split]
            _split_mols_file="molecules_split_{}.txt".format(split_code)
            _split_mols_file = os.path.join(args.split_info_dir,_split_mols_file)
            with open(_split_mols_file,"w") as _f:
                for _m in data:
                    _f.write(_m+"\n")
            try:
                proc_split(data,_of)
            except:
                traceback.print_exc()
                time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
                m = "{} {} {} {}. {}".format("="*100,"error in split",split_id,split_code,time_now)
                print(m)
                logging.exception(m)
                raise
                #print("="*100,"error in split",split_id,split_code)
            split_id+=1
        time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        m = "Done. {}".format(time_now)
        print(m)
        logging.info(m)

