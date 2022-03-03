
def config_JointEmbeder():   
    conf = {
        # data_params
        'dataset_name':'CodeSearchDataset',
		
        #training data
        'train_name':'train.name.h5',
        'train_tokens':'train.tokens.h5',
        'train_graphseq':'train.graphseq.h5',
        'train_desc':'train.desc.h5',
		
        #test data
        'valid_name':'valid.name.h5',
        'valid_tokens':'valid.tokens.h5',
        'valid_graphseq':'valid.graphseq.h5',
        'valid_desc':'valid.desc.h5',
		
        #use data (computing code vectors)
        'use_codebase':'use.rawcode.txt',
        'use_names':'use.name.h5',
        'use_tokens':'use.tokens.h5',
        'use_graphseq':'use.graphseq.h5',
		
        #results data(code vectors)            
        'use_codevecs':'use.codevecs.h5',        
			   
        #parameters
        'name_len': 6,
        'tokens_len':50,
        'graphseq_len': 80,
        'desc_len': 30,    
        'n_words': 10000,
		
        #vocabulary info
        'vocab_name':'vocab.name.json',
        'vocab_tokens':'vocab.tokens.json',
        'vocab_desc':'vocab.desc.json',
        'vocab_graphseq':'vocab.graphseq.json', 
		
        #training_params            
        'batch_size':64,
        'chunk_size':200000,
        'nb_epoch': 2001,
		
        #'optimizer': 'adam',
        'learning_rate':2.08e-4,
        'adam_epsilon':1e-8,
        'warmup_steps':5000,
        'fp16': False,
        'fp16_opt_level': 'O1', 

        # model_params
        'emb_size': 512,
        'n_hidden': 512,
        'lstm_dims': 256,      
        'margin': 0.3986,
        'sim_measure':'cos',
    }
    return conf



