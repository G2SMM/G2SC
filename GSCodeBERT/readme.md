# Code Search

## Data Process

- The dataset is randomly shuffled, the proportion of training set, validation set and test set is 8:1:1.

- In the train and valid dataset, positive and negative samples are balanced.For a positive sample <NL,PL>, the negative sample is chosen at random from the rest of the dataset, such as <NL, PL'>.

- For test dataset, positive sample : negative sample is 1:999

- The processed datasets is in the data fold.

  train.json, valid.json, test.json is the dataset after 8:1:1 split, only with positive samples

  ```json
  // 1. train/valid json dataset file description
  "url":sequence number; "code":code sequence; "docstring":natural language summaration for program;
  "fun_name":is not used,make no difference;"dfs":DFSCode of code 
  
  // 2.ps_train.txt/ps.valid.txt file description
  (label, index, index, description, code) is splited with <CODESPLIT>
  
  // 3.desc_dfs_train.txt/desc_dfs_valid.txt file description
  (label, index, index, description, dfscode) is splited with <CODESPLIT>
  ```

​         **process_data.py bulid()** function is used for build train/valid dataset negative samples;

​		 **process_test_data.py** build final test datasets.

​         In data folder, **desc_dfs_train.txt/ desc_dfs_train.txt** is used for Fine-Tune with G2SC, **ps_train.txt/ps_valid.txt** is used   for  CodeSearch Fine-Tune .

​        

## Fine-Tune with G2SC

We further trained the model after adding G2SC data

```sh
./run_desc_dfs.sh
```

```sh
pretrained_model=./codebert-base

python3 run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file desc_dfs_train.txt \
--dev_file desc_dfs_valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ./data/ \
--output_dir ./desc_dfs/  \
--model_name_or_path $pretrained_model
```

## CodeSearch Fine-Tune 

```sh
./run_code_desc.sh
```

```sh
pretrained_model=./desc_dfs/checkpoint-best

python3 run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file ps_train.txt \
--dev_file ps_valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ./data/ \
--output_dir ./code_desc/  \
--model_name_or_path $pretrained_model

```

## Infer

```sh
./run_test.sh
```

```sh
for ((i=0; i<=39; i++))
do
    python3 run_classifier.py \
    --model_type roberta \
    --model_name_or_path ./codebert-base \
    --task_name codesearch \
    --do_predict \
    --output_dir ./code_desc/ \
    --data_dir ./test/ \
    --max_seq_length 200 \
    --per_gpu_train_batch_size 128 \
    --per_gpu_eval_batch_size 128 \
    --learning_rate 1e-5 \
    --num_train_epochs 8 \
    --test_file batch_${i}.txt \
    --pred_model_dir ./code_desc/checkpoint-best/ \
    --test_result_dir ./results/$lang/${i}_batch_result.txt
done
```

## Evaluation

```sh
python mrr.py
```

