nvidia-smi


MAX_PIXELS=401408 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model xxxxxx \
    --train_type full \
    --model_type qwen2_5_vl \
    --dataset xxxxx \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --attn_impl flash_attn \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --save_steps 5000 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir xxxxxxx \
    --warmup_ratio 0.03 \
    --deepspeed zero2 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --eval_strategy no \
    --report_to wandb \
    --save_only_model true 
