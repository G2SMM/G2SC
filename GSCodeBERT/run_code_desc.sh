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
--output_dir ./code_desc/ \
--model_name_or_path $pretrained_model
