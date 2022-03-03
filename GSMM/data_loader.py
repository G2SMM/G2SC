import sys
import torch 
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle
from utils import PAD_ID, SOS_ID, EOS_ID, UNK_ID, indexes2sent

    
class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, data_dir, f_name, max_name_len, f_tokens, max_tok_len, f_graphseq, max_graphseq_len, f_descs=None, max_desc_len=None):
        self.max_name_len=max_name_len
        self.max_tok_len=max_tok_len
        self.max_graphseq_len=max_graphseq_len
        self.max_desc_len=max_desc_len
        # 1. Initialize file path or list of file names.
        """read training data(list of int arrays) from a hdf5 file"""
        self.training=False
        print("loading data...")
        table_name = tables.open_file(data_dir+f_name)
        self.names = table_name.get_node('/phrases')[:].astype(np.long)
        self.idx_names = table_name.get_node('/indices')[:]
		
        table_tokens = tables.open_file(data_dir+f_tokens)
        self.tokens = table_tokens.get_node('/phrases')[:].astype(np.long)
        self.idx_tokens = table_tokens.get_node('/indices')[:]
		
        table_graphseq = tables.open_file(data_dir+f_graphseq)
        self.graphseq = table_graphseq.get_node('/phrases')[:].astype(np.long)
        self.idx_graphseq = table_graphseq.get_node('/indices')[:]
        if f_descs is not None:
            self.training=True
            table_desc = tables.open_file(data_dir+f_descs)
            self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
            self.idx_descs = table_desc.get_node('/indices')[:]
        
        assert self.idx_names.shape[0] == self.idx_graphseq.shape[0]
        assert self.idx_graphseq.shape[0] == self.idx_tokens.shape[0]
        if f_descs is not None:
            assert self.idx_names.shape[0]==self.idx_descs.shape[0]
        self.data_len = self.idx_names.shape[0]
        print("{} entries".format(self.data_len))
        
    def pad_seq(self, seq, maxlen):
        if len(seq)<maxlen:
            seq=np.append(seq, [PAD_ID]*(maxlen-len(seq)))
        seq=seq[:maxlen]
        return seq
    
    def __getitem__(self, offset):          
        len, pos = self.idx_names[offset]['length'], self.idx_names[offset]['pos']
        name_len=min(int(len),self.max_name_len) 
        name = self.names[pos: pos+name_len]
        name = self.pad_seq(name, self.max_name_len)
        
        len, pos = self.idx_tokens[offset]['length'], self.idx_tokens[offset]['pos']
        tok_len = min(int(len), self.max_tok_len)
        tokens = self.tokens[pos:pos+tok_len]
        tokens = self.pad_seq(tokens, self.max_tok_len)
        
        len, pos = self.idx_graphseq[offset]['length'], self.idx_graphseq[offset]['pos']
        graphseq_len = min(int(len), self.max_graphseq_len)
        graphseq = self.graphseq[pos:pos+graphseq_len]
        graphseq= self.pad_seq(graphseq, self.max_graphseq_len)
        if self.training:
            len, pos = self.idx_descs[offset]['length'], self.idx_descs[offset]['pos']
            good_desc_len = min(int(len), self.max_desc_len)
            good_desc = self.descs[pos:pos+good_desc_len]
            good_desc = self.pad_seq(good_desc, self.max_desc_len)
            
            rand_offset=random.randint(0, self.data_len-1)
            len, pos = self.idx_descs[rand_offset]['length'], self.idx_descs[rand_offset]['pos']
            bad_desc_len=min(int(len), self.max_desc_len)
            bad_desc = self.descs[pos:pos+bad_desc_len]
            bad_desc = self.pad_seq(bad_desc, self.max_desc_len)

            return name, name_len, tokens, tok_len, graphseq, graphseq_len, good_desc, good_desc_len, bad_desc, bad_desc_len
        return name, name_len, tokens, tok_len, graphseq, graphseq_len
        
    def __len__(self):
        return self.data_len
    
def load_dict(filename):
    return json.loads(open(filename, "r").readline())

def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs= h5f.root.vecs
    
    vecs=np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:]=h5vecs[:]
    h5f.close()
    return vecs
        
def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()