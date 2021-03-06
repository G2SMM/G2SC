# Graph-to-Sequence enabled multi-modal model

PyTorch implementation of [GSMM]

## Dependency
> Tested in Ubuntu 16.04.6
* Python 3.6.2
* PyTorch 1.8.0
* tqdm 4.59.0
* tables 3.6.1
* nltk 3.5

## Code Structures

 - `models`: neural network models for code/desc representation and similarity measure.
 - `modules.py`: basic modules for model construction.
 - `train.py`: train and validate code/desc representaton models; 
 - `repr_code.py`: encode code into vectors and store them to a file; 
 - `search.py`: perform code search; 
 - `txt_search_query.py`:use txt document to perform code search 
 - `auto_eval.py`: automatic evaluation of code search model
 - `configs.py`: configurations for models defined in the `models` folder. 

   Each function defines the hyper-parameters for the corresponding model.
 - `data_loader.py`: A PyTorch dataset loader.
 - `utils.py`: utilities for models and training. 


## Usage 

   ### Data Preparation 
  The `/data` folder provides a small dummy dataset for quick deployment.  
  To train and test our model:
  
  1) Download and unzip real dataset from [Google Drive](https://drive.google.com/file/d/1pAjFDtnMJZC8uIktN4tFtZ7EbHsMA12L/view?usp=sharing)
  
  2) Replace each file in the `/data` folder with the corresponding real file. 
  
   ### Configuration
   Edit hyper-parameters and settings in `config.py`

   ### Train
   
   ```bash
   python train.py --model JointEmbeder 
   ```
   
   ### Code Embedding
   
   ```bash
   python repr_code.py --model JointEmbeder --reload_from XXX
   ```
   where `XXX` represents the iteration with the best model.
   
   For the step10000.h5 model file, we use "python repr_code.py --model JointEmbeder --reload_from 10000"
   
   ### Search
   
   ```bash
   python search.py --model JointEmbeder --reload_from XXX
   ```
   where `XXX` represents the iteration with the best model.
   
   For the step10000.h5 model file, we use "python search.py --model JointEmbeder --reload_from 10000"
   
   ### Search with txt document
   
   ```bash
   python txt_search_query.py --model JointEmbeder --reload_from XXX
   ```
   where `XXX` represents the iteration with the best model,and the query document is stored in `/data` folder
   
   For the step10000.h5 model file, we use "python txt_search_query.py --model JointEmbeder --reload_from 10000"
   
   ### Automatic evaluation
   
   ```bash
   python auto_eval.py --model JointEmbeder --reload_from XXX
   ```
   where `XXX` represents the iteration with the best model.
   
   For the step10000.h5 model file, we use "python auto_eval.py --model JointEmbeder --reload_from 10000"
 
