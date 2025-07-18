python qlora.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --dataset 'fine_tune_data.json'\
    --dataset_format alpaca\
    --use_auth \
    --output_dir ./output/llama-2-schedule-llm-13b \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_dataset_size 500 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 8 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_mmlu_eval False\
    --do_eval False\
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --source_max_len 1024 \
    --target_max_len 1024 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_steps 5000 \
    --eval_steps 1000 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --report_to wandb \
    --seed 0 \