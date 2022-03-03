import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)
from modules import SeqEncoder, BOWEncoder

class JointEmbeder(nn.Module):

    def __init__(self, config):
        super(JointEmbeder, self).__init__()
        self.conf = config
        self.margin = config['margin']
        self.hidden=config['n_hidden']
		
        self.name_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        self.tok_encoder=BOWEncoder(config['n_words'],config['emb_size'],config['n_hidden'])
        self.graphseq_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        self.desc_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
                          
        self.w_name = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        self.w_tok = nn.Linear(config['emb_size'], config['n_hidden'])
        self.w_graphseq=nn.Linear(2*config['lstm_dims'], config['n_hidden'])  
        self.w_desc = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        
        self.w_atten = nn.Linear(config['n_hidden'], 1)
        
        self.fuse = nn.Linear(config['n_hidden']*3,config['n_hidden'])
        self.init_weights()
        
    def init_weights(self):# Initialize Linear Weight 
        for m in [self.w_name, self.w_tok, self.w_graphseq, self.w_desc, self.w_atten, self.fuse]:        
            m.weight.data.uniform_(-0.1, 0.1)#nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.) 
            
    def code_encoding(self, name, name_len, tokens, tok_len, graphseq, graphseq_len):
        batch_size=name.shape[0]
		
        name_repr=self.name_encoder(name, name_len)
        tok_repr=self.tok_encoder(tokens, tok_len)
        graphseq_repr=self.graphseq_encoder(graphseq, graphseq_len)
		
        name_feat_hidden = self.w_name(name_repr).reshape(-1, self.hidden)
        tok_feat_hidden = self.w_tok(tok_repr).reshape(-1, self.hidden)
        graphseq_feat_hidden = self.w_graphseq(graphseq_repr).reshape(-1, self.hidden)
        
        name_attn_tanh = torch.tanh(name_feat_hidden)
        name_attn_scalar = self.w_atten(F.dropout(name_attn_tanh, 0.25 ,self.training).reshape(-1, self.hidden))            
        
        tok_attn_tanh = torch.tanh(tok_feat_hidden)
        tok_attn_scalar = self.w_atten(F.dropout(tok_attn_tanh, 0.25 ,self.training).reshape(-1, self.hidden))
        
        graphseq_attn_tanh = torch.tanh(graphseq_feat_hidden)
        graphseq_attn_scalar = self.w_atten(F.dropout(graphseq_attn_tanh, 0.25 ,self.training).reshape(-1, self.hidden))
        
        attn_cat = torch.cat([name_attn_scalar, tok_attn_scalar, graphseq_attn_scalar], 1)
        atten_weight = F.softmax(attn_cat, dim=1)
        
        name_feat_atten = torch.bmm(atten_weight[:,0].reshape(batch_size, 1, 1),name_repr.reshape(batch_size, 1, self.hidden))     
        tok_feat_atten = torch.bmm(atten_weight[:,0].reshape(batch_size, 1, 1),tok_repr.reshape(batch_size, 1, self.hidden))
        graphseq_feat_atten = torch.bmm(atten_weight[:,0].reshape(batch_size, 1, 1),graphseq_repr.reshape(batch_size, 1, self.hidden))
        cat_atten_repr = torch.cat((name_feat_atten, tok_feat_atten, graphseq_feat_atten), 2)
        code_repr = torch.tanh(self.fuse(F.dropout(cat_atten_repr, 0.25, training=self.training))).reshape(-1,self.hidden)
        return code_repr
        
    def desc_encoding(self, desc, desc_len):
        desc_repr=self.desc_encoder(desc, desc_len)
        desc_repr=self.w_desc(desc_repr)
        return desc_repr
    
    def similarity(self, code_vec, desc_vec):
        assert self.conf['sim_measure'] in ['cos', 'poly', 'euc', 'sigmoid', 'gesd', 'aesd'], "invalid similarity measure"
        if self.conf['sim_measure']=='cos':
            return F.cosine_similarity(code_vec, desc_vec)
        elif self.conf['sim_measure']=='poly':
            return (0.5*torch.matmul(code_vec, desc_vec.t()).diag()+1)**2
        elif self.conf['sim_measure']=='sigmoid':
            return torch.tanh(torch.matmul(code_vec, desc_vec.t()).diag()+1)
        elif self.conf['sim_measure'] in ['euc', 'gesd', 'aesd']:
            euc_dist = torch.dist(code_vec, desc_vec, 2) # or torch.norm(code_vec-desc_vec,2)
            euc_sim = 1 / (1 + euc_dist)
            if self.conf['sim_measure']=='euc': return euc_sim                
            sigmoid_sim = torch.sigmoid(torch.matmul(code_vec, desc_vec.t()).diag()+1)
            if self.conf['sim_measure']=='gesd': 
                return euc_sim * sigmoid_sim
            elif self.conf['sim_measure']=='aesd':
                return 0.5*(euc_sim+sigmoid_sim)
    
    def forward(self, name, name_len, tokens, tok_len, graphseq, graphseq_len, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):
        batch_size=name.size(0)
        code_repr=self.code_encoding(name, name_len, tokens, tok_len, graphseq, graphseq_len)
        desc_anchor_repr=self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr=self.desc_encoding(desc_neg, desc_neg_len)
    
        anchor_sim = self.similarity(code_repr, desc_anchor_repr)
        neg_sim = self.similarity(code_repr, desc_neg_repr) # [batch_sz x 1]
        
        loss=(self.margin-anchor_sim+neg_sim).clamp(min=1e-6).mean()
        
        return loss
