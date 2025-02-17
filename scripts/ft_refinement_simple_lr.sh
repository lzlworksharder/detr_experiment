#!/bin/bash
python ft_simple_lr.py \
    --dataset_name_or_path ./cppe-5\
    --image_size 480\
    --per_device_train_batch_size 16\
    --per_device_eval_batch_size 32\
    --preprocessing_num_workers 2\
    --model_name_or_path ./models/SenseTime/deformable-detr-with-box-refine\
    --learning_rate 1e-4\
    --score_threshold 0.0\
    --num_warmup_steps 0\
    --weight_decay 1e-4\
    --num_train_epochs 50\
    --logging_steps 10\
    --gradient_accumulation_steps 1\
    --logging_dir ./log\
    --project_name ./refine_simple_lr\
    --output_dir ./refine_out_simple_lr\
    --adam_beta1 0.9\
    --adam_beta2 0.999\
    --adam_epsilon 1e-8\








