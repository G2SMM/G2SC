import os
import sys
import random
import time
from datetime import datetime
import numpy as np
import math
import argparse
random.seed(42)
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from tensorboardX import SummaryWriter

import torch
import models.jointemb,configs, data_loader 
from modules import get_cosine_schedule_with_warmup
from utils import similarity, normalize
from data_loader import *

try: 
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML, SESSION_NAME
except: 
    IS_ON_NSML = False
    
def bind_nsml(model, **kwargs):
    if type(model) == torch.nn.DataParallel: model = model.module
    def infer(raw_data, **kwargs):
        pass
    def load(path, *args):
        weights = torch.load(path)
        model.load_state_dict(weights)
        logger.info(f'Load checkpoints...!{path}')
    def save(path, *args):
        torch.save(model.state_dict(), os.path.join(path, 'model.pkl'))
        logger.info(f'Save checkpoints...!{path}')
    nsml.bind(save, load, infer)

    
def train(args):
    fh = logging.FileHandler(f"./output/{args.model}/{args.dataset}/logs.txt")                                   
    logger.addHandler(fh)
    timestamp = datetime.now().strftime('%Y%m%d%H%M') 
    tb_writer = SummaryWriter(f"./output/{args.model}/{args.dataset}/logs/{timestamp}" ) if args.visual else None
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu") 

    config=getattr(configs, 'config_'+args.model)()
    if args.automl:
        config.update(vars(args))
    print(config)
    
    ###############################################################################
    # Load data
    ###############################################################################
    data_path = DATASET_PATH+"/train/" if IS_ON_NSML else args.data_path+args.dataset+'/'
    train_set = eval(config['dataset_name'])(data_path, 
	                          config['train_name'], config['name_len'],
                                  config['train_tokens'], config['tokens_len'],
                                  config['train_graphseq'], config['graphseq_len'],
				  config['train_desc'], config['desc_len'])
    valid_set = eval(config['dataset_name'])(data_path,
                                  config['valid_name'], config['name_len'],
                                  config['valid_tokens'], config['tokens_len'],
                                  config['valid_graphseq'], config['graphseq_len'],
	                          config['valid_desc'], config['desc_len'])
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'],shuffle=True, drop_last=True, num_workers=1)
    
    ###############################################################################
    # Define Model
    ###############################################################################
    logger.info('Constructing Model..')
    model = getattr(models.jointemb, args.model)(config)#initialize the model
    
    def save_model(model, ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    def load_model(model, ckpt_path, to_device):
        assert os.path.exists(ckpt_path), f'Weights not found'
        model.load_state_dict(torch.load(ckpt_path, map_location=to_device))
        
    if args.reload_from>0:
        ckpt = f'./output/{args.model}/{args.dataset}/models/step{args.reload_from}.h5'
        load_model(model, ckpt, device)    
        
    if IS_ON_NSML:
        bind_nsml(model)
    model.to(device)    
    
    ###############################################################################
    # Prepare the Optimizer
    ###############################################################################
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])        
    scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=config['warmup_steps'], 
            num_training_steps=len(data_loader)*config['nb_epoch']) # do not foget to modify the number when dataset is changed
    if config['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=config['fp16_opt_level'])
    
    ###############################################################################
    # Training Process
    ###############################################################################    
    n_iters = len(data_loader)
    itr_global = args.reload_from+1 
    for epoch in range(int(args.reload_from/n_iters)+1, config['nb_epoch']+1): 
        itr_start_time = time.time()
        losses=[]
        for batch in data_loader:
            
            model.train()
            batch_gpu = [tensor.to(device) for tensor in batch]
            loss = model(*batch_gpu)
            
            if config['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            losses.append(loss.item())
            
            if itr_global % args.log_every ==0:
                elapsed = time.time() - itr_start_time
                logger.info('epo:[%d/%d] itr:[%d/%d] step_time:%ds Loss=%.5f'%
                        (epoch, config['nb_epoch'], itr_global%n_iters, n_iters, elapsed, np.mean(losses)))
                if tb_writer is not None:
                    tb_writer.add_scalar('loss', np.mean(losses), itr_global)
                if IS_ON_NSML:
                    summary = {"summary": True, "scope": locals(), "step": itr_global}
                    summary.update({'loss':np.mean(losses)})
                    nsml.report(**summary)
                    
                losses=[] 
                itr_start_time = time.time() 
            itr_global = itr_global + 1

            if itr_global % args.valid_every == 0:
                logger.info("validating..")                  
                valid_result = validate(valid_set, model, 1000, 10, config['sim_measure'])  
                logger.info(valid_result)
                if tb_writer is not None:
                    for key, value in valid_result.items():
                        tb_writer.add_scalar(key, value, itr_global)
                if IS_ON_NSML:
                    summary = {"summary": True, "scope": locals(), "step": itr_global}
                    summary.update(valid_result)
                    nsml.report(**summary)
                    
            if itr_global % args.save_every == 0:
                ckpt_path = f'./output/{args.model}/{args.dataset}/models/step{itr_global}.h5'
                save_model(model, ckpt_path)
                if IS_ON_NSML:
                    nsml.save(checkpoint=f'model_step{itr_global}')
            if itr_global==310000:
                sys.exit(0)
##### Evaluation #####
def validate(valid_set, model, pool_size, K, sim_measure):
    def ACC(real,predict):
        sum=0.0
        for val in real:
            try: index=predict.index(val)
            except ValueError: index=-1
            if index!=-1: sum=sum+1  
        return sum/float(len(real))
    def MAP(real,predict):
        sum=0.0
        for id, val in enumerate(real):
            try: index=predict.index(val)
            except ValueError: index=-1
            if index!=-1: sum=sum+(id+1)/float(index+1)
        return sum/float(len(real))
    def MRR(real, predict):
        sum=0.0
        for val in real:
            try: index = predict.index(val)
            except ValueError: index=-1
            if index!=-1: sum=sum+1.0/float(index+1)
        return sum/float(len(real))
    def NDCG(real, predict):
        dcg=0.0
        idcg=IDCG(len(real))
        for i, predictItem in enumerate(predict):
            if predictItem in real:
                itemRelevance = 1
                rank = i+1
                dcg +=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(rank+1))
        return dcg/float(idcg)
    def IDCG(n):
        idcg=0
        itemRelevance=1
        for i in range(n): idcg+=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(i+2))
        return idcg

    model.eval()
    device = next(model.parameters()).device

    data_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=10000, 
                                 shuffle=True, drop_last=True, num_workers=1)
    accs, mrrs, maps, ndcgs=[],[],[],[]
    code_reprs, desc_reprs = [], []
    n_processed = 0
    for batch in tqdm(data_loader):        
        if len(batch) == 10:
            code_batch = [tensor.to(device) for tensor in batch[:6]]
            desc_batch = [tensor.to(device) for tensor in batch[6:8]]
        with torch.no_grad():
            code_repr=model.code_encoding(*code_batch).data.cpu().numpy().astype(np.float32)
            desc_repr=model.desc_encoding(*desc_batch).data.cpu().numpy().astype(np.float32) # [poolsize x hid_size]
            if sim_measure=='cos':
                code_repr = normalize(code_repr)
                desc_repr = normalize(desc_repr)
        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        n_processed += batch[0].size(0)
    code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)
     
    for k in tqdm(range(0, n_processed, pool_size)):
        code_pool, desc_pool = code_reprs[k:k+pool_size], desc_reprs[k:k+pool_size] 
        for i in range(min(10000, pool_size)):
            desc_vec = np.expand_dims(desc_pool[i], axis=0) 
            n_results = K    
            if sim_measure=='cos':
                sims = np.dot(code_pool, desc_vec.T)[:,0] 
            else:
                sims = similarity(code_pool, desc_vec, sim_measure)              
            negsims=np.negative(sims)
            predict = np.argpartition(negsims, kth=n_results-1)
            predict = predict[:n_results]   
            predict = [int(k) for k in predict]
            real = [i]
            accs.append(ACC(real,predict))
            mrrs.append(MRR(real,predict))
            maps.append(MAP(real,predict))
            ndcgs.append(NDCG(real,predict))                     
    return {'acc':np.mean(accs), 'mrr': np.mean(mrrs), 'map': np.mean(maps), 'ndcg': np.mean(ndcgs)}   
    
def parse_args():
    parser = argparse.ArgumentParser("Train and Validate The Code Search (Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('--dataset', type=str, default='github', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')
   
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual",action="store_true", default=False, help="Visualize training status in tensorboard")
    parser.add_argument('--automl', action='store_true', default=False, help='use automl')
    # Training Arguments
    parser.add_argument('--log_every', type=int, default=100, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=5000, help='interval to validation')
    parser.add_argument('--save_every', type=int, default=10000, help='interval to evaluation to concrete results')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
        
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument('--n_hidden', type=int, default= -1, help='number of hidden dimension of code/desc representation')
    parser.add_argument('--lstm_dims', type=int, default= -1)         
    parser.add_argument('--margin', type=float, default= -1)
    parser.add_argument('--sim_measure', type=str, default = 'cos', help='similarity measure for training')
    
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    
    # reserved args for automl pbt
    parser.add_argument('--pause', default=0, type=int)
    parser.add_argument('--iteration', default=0, type=str)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()   
    # make output directory if it doesn't already exist
    os.makedirs(f'./output/{args.model}/{args.dataset}/models', exist_ok=True)
    os.makedirs(f'./output/{args.model}/{args.dataset}/tmp_results', exist_ok=True)
    
    torch.backends.cudnn.benchmark = True # speed up training by using cudnn
    torch.backends.cudnn.deterministic = True # fix the random seed in cudnn
    train(args)
        
    
