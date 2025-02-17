
import argparse
import math
import os
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed,ProjectConfiguration
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
from peft import LoraConfig, get_peft_model
from pprint import pprint
from transformers import (
    AutoImageProcessor,
    get_scheduler,
    AutoModelForObjectDetection
)
from PIL import Image,ImageDraw
import albumentations as A 
from functools import partial
import utils
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import LambdaLR

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for model training.")

    # data
    # load data:
    parser.add_argument('--dataset_name_or_path', type=str, help='Dataset name', default=r'src_data\rajpurkar\squad')

    # pre-processing
    parser.add_argument('--image_size',type=int,default=480,help='image width&height used in preprocessing')
    # dataloader: batchsize + num_workers + num_samples
    parser.add_argument('--max_train_samples', type=int, help='Max train samples', default=None)
    parser.add_argument('--max_validation_samples', type=int, help='Max validation samples', default=None)
    parser.add_argument('--per_device_train_batch_size', type=int, help='Per device train batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, help='Per device eval batch size', default=8)
    parser.add_argument('--preprocessing_num_workers', type=int, help='Number of workers', default=2)

    # model
    # load model
    parser.add_argument('--model_name_or_path', type=str, help='Model name or path')
    parser.add_argument('--lora_r',type=int,default=None)
    parser.add_argument('--lora_alpha',type=int,default=None)
    parser.add_argument('--lora_modules_to_save',nargs='*',type=str,default=None)
    parser.add_argument('--lora_dropout',type=float,default=0.0)
    parser.add_argument('--lora_target_modules',nargs='*',type=str,default=['q_proj','v_proj'])

    # training
    # optimizer(lr + wd) & scheduler(type + warmup)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=5e-5)
    parser.add_argument('--weight_decay', type=float, help='Weight decay', default=0)
    parser.add_argument('--lr_scheduler_type', type=str, help='Learning rate scheduler type', default='linear')
    parser.add_argument('--num_warmup_steps', type=int, help='Number of warmup steps', default=0)
    parser.add_argument('--score_threshold',type=float,default=0.0)
    parser.add_argument('--adam_beta1',type=float,default=0.9)
    parser.add_argument('--adam_beta2',type=float,default=0.999)
    parser.add_argument('--adam_epsilon',type=float,default=1e-8)
    # trainer: epochs, accumulation, precision
    parser.add_argument('--num_train_epochs', type=int, help='Number of training epochs', default=3)
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps', default=1)
    parser.add_argument('--mixed_precision', type=str, help='Mixed precision', default='fp16')

    # metrics & logging
    parser.add_argument('--logging_dir', type=str, help='Logging directory', default='./log')
    parser.add_argument('--report_to', type=str, help='Reporting method', default='tensorboard')
    parser.add_argument('--with_tracking', type=bool, help='Whether to track', default=True)
    parser.add_argument('--logging_steps', type=int, help='Logging steps', default=5)
    parser.add_argument('--project_name', type=str, help='Project name',default='project')

    # ckpt
    parser.add_argument('--save_best', action='store_true', help='Whether to save the best model')
    parser.add_argument('--output_dir', type=str, help='Output directory',default='./output')
    parser.add_argument('--checkpointing_steps', type=str, help='Checkpointing steps', default='epoch')

    # seed
    parser.add_argument('--seed', type=int, help='Random seed', default=42)

    # 返回解析后的参数
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    # accelerator
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        config = ProjectConfiguration(project_dir='./',logging_dir=args.logging_dir)
        accelerator_log_kwargs["project_config"] = config
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, 
                            mixed_precision=args.mixed_precision,
                            **accelerator_log_kwargs
                            )
    if args.with_tracking:
        accelerator.init_trackers(args.project_name)

    # load data & pre-processing
    cppe5 = load_dataset(args.dataset_name_or_path)
    if 'validation' not in cppe5:
        split = cppe5['train'].train_test_split(0.15,seed=args.seed)
        cppe5['train'] = split['train']
        cppe5['validation'] = split['test']
    if args.max_train_samples:
        cppe5['train'] = cppe5['train'].select(range(args.max_train_samples))
    if args.max_validation_samples:
        cppe5['validation'] = cppe5['validation'].select(range(args.max_validation_samples))
    categories = cppe5["train"].features["objects"].feature["category"].names
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        do_resize=True,
        size={'longest_edge':args.image_size,'shortest_edge':args.image_size},
        do_pad=True,
        pad_size ={'height':args.image_size,'width':args.image_size}, 
        
    )

    train_augment_and_transform = A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(format='coco',label_fields=['category'],clip=True,min_area=25)
    )

    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params = A.BboxParams(format='coco',label_fields=['category'],clip=True)
    )
    train_transform_batch = partial(
        utils.augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )
    validation_transform_batch = partial(
        utils.augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
    )

    train_set = cppe5["train"].with_transform(train_transform_batch)
    validation_set = cppe5["validation"].with_transform(validation_transform_batch)

    # dataloader
    def collate_fn(batch):
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        if "pixel_mask" in batch[0]:
            data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
        return data
    train_dataloader = DataLoader(
        train_set,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        # num_workers=args.preprocessing_num_workers,
        )
    validation_dataloader = DataLoader(
        validation_set,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
        # num_workers=args.preprocessing_num_workers
        )

    # load model
    model = AutoModelForObjectDetection.from_pretrained(
    args.model_name_or_path,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
    if args.lora_r is not None:
        config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha if args.lora_alpha is not None else 16,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        modules_to_save=args.lora_modules_to_save,
    )
        model = get_peft_model(model, config)
    # training prep: optimizer/scheduler
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,betas=(args.adam_beta1,args.adam_beta2),eps=args.adam_epsilon)


    lr_scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: 1 if step < int(0.8*args.num_train_epochs*len(train_dataloader)) else 0.1  
)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
    )
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)


    # training loop
    completed_steps = 0
    best_eval = 0 if args.save_best else None
    utils.print_model_info(model)
    for epoch in range(args.num_train_epochs):
        pbar = tqdm(
            range(math.ceil(len(train_dataloader)/args.gradient_accumulation_steps)),
            desc=f'Epoch {epoch+1}/{args.num_train_epochs}',
            leave=True if epoch+1==args.num_train_epochs else False,
            position=0,
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process
            )
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                try:
                    outputs = model(**batch)
                except ValueError:
                    print('nan error')
                    continue
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # logging
            if accelerator.sync_gradients:
                pbar.update(1)
                completed_steps += 1
                if args.with_tracking:
                    if completed_steps % args.logging_steps == 0:
                        train_log = {
                            'tr_Loss': loss.item(),
                            'lr': optimizer.param_groups[0]['lr'],
                            }
                        accelerator.log(train_log,step=completed_steps)
                        pbar.set_postfix(train_log)

            # ckpt
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

        if args.checkpointing_steps == "epoch":
            output_dir= args.output_dir if args.output_dir is not None else './output'
            accelerator.save_state(output_dir)
            

        # Evaluation
        model.eval()
        val_pbar = tqdm(
            range(len(validation_dataloader)),
            desc=f'validation',
            leave=False,
            disable=not accelerator.is_local_main_process,
            position=1,
            dynamic_ncols=True,
            )

        metric = MeanAveragePrecision(box_format='xyxy',class_metrics=True)


        for idx,batch in enumerate(validation_dataloader):
            pixel_values,labels = batch['pixel_values'],batch['labels']
            with torch.inference_mode():
                outputs = model(pixel_values)
            post_outputs = image_processor.post_process_object_detection(outputs,
                                                                        target_sizes=[d['orig_size'] for d in labels],
                                                                        threshold=args.score_threshold)
            for d in labels:
                d['labels'] = d.pop('class_labels')
                d['boxes'] = utils.convert_bbox_yolo_to_pascal(d['boxes'],d['orig_size'])

            metric.update(post_outputs,labels)
            torch.cuda.empty_cache()
            val_pbar.update(1)
        val_pbar.close()
        eval_metric =  metric.compute()

        classes = eval_metric.pop("classes")
        map_per_class = eval_metric.pop("map_per_class")
        mar_100_per_class = eval_metric.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
            eval_metric[f"map_{class_name}"] = class_map
            eval_metric[f"mar_100_{class_name}"] = class_mar
        eval_metric = {k: round(v.item(), 4) for k, v in eval_metric.items()}
        pprint(eval_metric)

        del outputs, loss, batch, metric  
        
        # logging & save best
        if args.save_best:
            if eval_metric['map'] > best_eval:
                best_eval = eval_metric['map']
                if args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
        if args.with_tracking:
            accelerator.log(eval_metric, step=completed_steps)
        pbar.close()


    # ckpt in the end
    if args.output_dir is not None and not args.save_best:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )

if __name__ == '__main__':
    main()
