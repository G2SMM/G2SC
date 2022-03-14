# Code Search

## Data Process

- The dataset is randomly shuffled, the proportion of training set, validation set and test set is 8:1:1.

- In the train and valid dataset, positive and negative samples are balanced.For a positive sample <NL,PL>, the negative sample is chosen at random from the rest of the dataset, such as <NL, PL'>.

- For test dataset, positive sample : negative sample is 1:999

- The processed datasets is in the data fold.

  train.json, valid.json, test.json is the dataset after 8:1:1 split, only with positive samples, these dataset don't contain graph sequences.  

  train_gs.json, valid_gs.json contain graph sequences, and delete some samples that the code can't convert to PDG 
  
  ```json
  // 1. train/valid json dataset file description
  "url":sequence number; "code":code sequence; "docstring":natural language summaration for program;
  "fun_name":is not used,make no difference;
  
  // 2. train_gs/valid_gs json dataset file description
  "url":sequence number; "code":code sequence; "docstring":natural language summaration for program;
  "fun_name":is not used,make no difference;"dfs":code graph sequence
  
  // 3.desc_code_train.txt/desc_code_valid.txt file description
  (label, index, index, description, code) is splited with <CODESPLIT>
  
  // 4.desc_gs_train.txt/desc_gs_valid.txt file description
  (label, index, index, description, dfscode) is splited with <CODESPLIT>
  ```

​         **process_data.py bulid()** function is used for build train/valid dataset negative samples

```py
###  desc_gs_train.txt/ desc_gs_train.txt is used for Fine-Tune with code graph sequence 
build('train_gs.json', 'desc_gs_train','code')
build('valid_gs.json', 'desc_gs_valid','code')
### desc_code_train.txt/desc_code_valid.txt is used for CodeSearch Fine-Tune
build('train.json', 'desc_code_train','dfs')
build('valid.json', 'desc_code_valid','dfs')
```

​		 **process_test_data.py** build final test datasets.

​        

For this step, run: 

```sh
python process_data.py bulid()
python process_test_data.py
```

## Fine-Tune with DFSCode

We further trained the model after adding DFScode data

```sh
bash run_desc_gs.sh
```

```sh
pretrained_model=./codebert-base

python3 run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file desc_gs_train.txt \
--dev_file desc_gs_valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ./ \
--output_dir ./desc_gs/  \
--model_name_or_path $pretrained_model
```

## CodeSearch Fine-Tune 

```bash
bash run_desc_code.sh
```

```sh
pretrained_model=./desc_gs/checkpoint-best

python3 run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file desc_code_train.txt \
--dev_file desc_code_valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ./ \
--output_dir ./desc_code/  \
--model_name_or_path $pretrained_model

```

## Infer

```sh
bash run_test.sh
```

```sh
for ((i=0; i<=39; i++))
do
    python3 run_classifier.py \
    --model_type roberta \
    --model_name_or_path ./codebert-base \
    --task_name codesearch \
    --do_predict \
    --output_dir ./desc_code/ \
    --data_dir ./test/test/java/ \
    --max_seq_length 200 \
    --per_gpu_train_batch_size 128 \
    --per_gpu_eval_batch_size 128 \
    --learning_rate 1e-5 \
    --num_train_epochs 8 \
    --test_file batch_${i}.txt \
    --pred_model_dir ./desc_code/checkpoint-best/ \
    --test_result_dir ./results/$lang/${i}_batch_result.txt
done
```

## Evaluation

```sh
python mrr.py
```

