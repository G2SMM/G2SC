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
    --test_result_dir ./results/${i}_batch_result.txt
done
