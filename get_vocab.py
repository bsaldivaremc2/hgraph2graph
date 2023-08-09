import sys
import argparse 
from hgraph import *
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool


def check_validity(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return False
    else:
        try:
            Chem.SanitizeMol(m)
            return True
        except:
            return False

def check_validity_batch(smi_list,otype="list"):
    if otype=="list":
        _o=[]
    else:
        _o = set()
    for smi in smi_list:
        if check_validity(smi):
            if otype=="list":
                _o.append(smi)
            else:
                _o.add(smi)
    return _o

def process(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
    return vocab

def parse_lines(lines):
    _o = []
    for line in lines:
        _l = line.strip("\r\n ")
        if len(_l)>0:
            _o.append(_l)
    return _o


def pool_proc(idata,ifunc,otype="list"):
    batch_size = len(idata) // args.ncpu + 1
    batches = [idata[i : i + batch_size] for i in range(0, len(idata), batch_size)]
    pool = Pool(args.ncpu)
    _result = pool.map(ifunc, batches)
    if otype=="list":
        o = []
    else:
        o = set()
    for _r in _result:
        if otype=="list":
            o.extend(_r)
        else:
            o.add(_r)
    return o

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=1)
    parser.add_argument('--outputfile', type=str, default="output_vocabulary.txt")
    parser.add_argument('--molecule_file', type=str, default="molecules_smiles.txt")
    args = parser.parse_args()

    print("Reading molecule list")
    with open(args.molecule_file,"r") as f:
        _ = f.read().split("\n")
    data = pool_proc(_,parse_lines,otype="list")
    data = list(set(data))
    print("Total molecules",len(data))
    print("Checking validity")
    data = pool_proc(data,check_validity_batch,otype="list")
    data = list(set(data))
    print("Mol list gotten:",len(data))
    print("Getting batches for vocab")
    batch_size = len(data) // args.ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    print("Pooling")
    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, batches)
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    vocab = list(set(vocab))
    print("Saving result in ",args.outputfile)
    with open(args.outputfile,"w") as f:
        for x,y in tqdm(sorted(vocab)):
            f.write("{} {}".format(x,y))
        #print(x, y)

